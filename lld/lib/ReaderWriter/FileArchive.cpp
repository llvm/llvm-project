//===- lib/ReaderWriter/FileArchive.cpp -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Driver/Driver.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <set>
#include <unordered_map>

using llvm::object::Archive;
using llvm::object::ObjectFile;
using llvm::object::SymbolRef;
using llvm::object::symbol_iterator;
using llvm::object::object_error;

namespace lld {

namespace {

/// \brief The FileArchive class represents an Archive Library file
class FileArchive : public lld::ArchiveLibraryFile {
public:
  FileArchive(std::unique_ptr<MemoryBuffer> mb, const Registry &reg,
              StringRef path, bool logLoading)
      : ArchiveLibraryFile(path), _mb(std::shared_ptr<MemoryBuffer>(mb.release())),
        _registry(reg), _logLoading(logLoading) {}

  /// \brief Check if any member of the archive contains an Atom with the
  /// specified name and return the File object for that member, or nullptr.
  File *find(StringRef name, bool dataSymbolOnly) override {
    auto member = _symbolMemberMap.find(name);
    if (member == _symbolMemberMap.end())
      return nullptr;
    Archive::child_iterator ci = member->second;
    if (ci->getError())
      return nullptr;

    // Don't return a member already returned
    ErrorOr<StringRef> buf = (*ci)->getBuffer();
    if (!buf)
      return nullptr;
    const char *memberStart = buf->data();
    if (_membersInstantiated.count(memberStart))
      return nullptr;
    if (dataSymbolOnly && !isDataSymbol(ci, name))
      return nullptr;

    _membersInstantiated.insert(memberStart);

    std::unique_ptr<File> result;
    if (instantiateMember(ci, result))
      return nullptr;

    File *file = result.get();
    _filesReturned.push_back(std::move(result));

    // Give up the file pointer. It was stored and will be destroyed with destruction of FileArchive
    return file;
  }

  /// \brief parse each member
  std::error_code
  parseAllMembers(std::vector<std::unique_ptr<File>> &result) override {
    if (std::error_code ec = parse())
      return ec;
    for (auto mf = _archive->child_begin(), me = _archive->child_end();
         mf != me; ++mf) {
      std::unique_ptr<File> file;
      if (std::error_code ec = instantiateMember(mf, file))
        return ec;
      result.push_back(std::move(file));
    }
    return std::error_code();
  }

  const AtomVector<DefinedAtom> &defined() const override {
    return _noDefinedAtoms;
  }

  const AtomVector<UndefinedAtom> &undefined() const override {
    return _noUndefinedAtoms;
  }

  const AtomVector<SharedLibraryAtom> &sharedLibrary() const override {
    return _noSharedLibraryAtoms;
  }

  const AtomVector<AbsoluteAtom> &absolute() const override {
    return _noAbsoluteAtoms;
  }

protected:
  std::error_code doParse() override {
    // Make Archive object which will be owned by FileArchive object.
    std::error_code ec;
    _archive.reset(new Archive(_mb->getMemBufferRef(), ec));
    if (ec)
      return ec;
    if ((ec = buildTableOfContents()))
      return ec;
    return std::error_code();
  }

private:
  std::error_code instantiateMember(Archive::child_iterator cOrErr,
                                    std::unique_ptr<File> &result) const {
    if (std::error_code ec = cOrErr->getError())
      return ec;
    Archive::child_iterator member = cOrErr->get();
    ErrorOr<llvm::MemoryBufferRef> mbOrErr = (*member)->getMemoryBufferRef();
    if (std::error_code ec = mbOrErr.getError())
      return ec;
    llvm::MemoryBufferRef mb = mbOrErr.get();
    std::string memberPath = (_archive->getFileName() + "("
                           + mb.getBufferIdentifier() + ")").str();

    if (_logLoading)
      llvm::errs() << memberPath << "\n";

    std::unique_ptr<MemoryBuffer> memberMB(MemoryBuffer::getMemBuffer(
        mb.getBuffer(), mb.getBufferIdentifier(), false));

    ErrorOr<std::unique_ptr<File>> fileOrErr =
        _registry.loadFile(std::move(memberMB));
    if (std::error_code ec = fileOrErr.getError())
      return ec;
    result = std::move(fileOrErr.get());
    if (std::error_code ec = result->parse())
      return ec;
    result->setArchivePath(_archive->getFileName());

    // The memory buffer is co-owned by the archive file and the children,
    // so that the bufffer is deallocated when all the members are destructed.
    result->setSharedMemoryBuffer(_mb);
    return std::error_code();
  }

  // Parses the given memory buffer as an object file, and returns true
  // code if the given symbol is a data symbol. If the symbol is not a data
  // symbol or does not exist, returns false.
  bool isDataSymbol(Archive::child_iterator cOrErr, StringRef symbol) const {
    if (cOrErr->getError())
      return false;
    Archive::child_iterator member = cOrErr->get();
    ErrorOr<llvm::MemoryBufferRef> buf = (*member)->getMemoryBufferRef();
    if (buf.getError())
      return false;
    std::unique_ptr<MemoryBuffer> mb(MemoryBuffer::getMemBuffer(
        buf.get().getBuffer(), buf.get().getBufferIdentifier(), false));

    auto objOrErr(ObjectFile::createObjectFile(mb->getMemBufferRef()));
    if (objOrErr.getError())
      return false;
    std::unique_ptr<ObjectFile> obj = std::move(objOrErr.get());

    for (SymbolRef sym : obj->symbols()) {
      // Skip until we find the symbol.
      ErrorOr<StringRef> name = sym.getName();
      if (!name)
        return false;
      if (*name != symbol)
        continue;
      uint32_t flags = sym.getFlags();
      if (flags <= SymbolRef::SF_Undefined)
        continue;

      // Returns true if it's a data symbol.
      SymbolRef::Type type = sym.getType();
      if (type == SymbolRef::ST_Data)
        return true;
    }
    return false;
  }

  std::error_code buildTableOfContents() {
    DEBUG_WITH_TYPE("FileArchive", llvm::dbgs()
                                       << "Table of contents for archive '"
                                       << _archive->getFileName() << "':\n");
    for (const Archive::Symbol &sym : _archive->symbols()) {
      StringRef name = sym.getName();
      ErrorOr<Archive::child_iterator> memberOrErr = sym.getMember();
      if (std::error_code ec = memberOrErr.getError())
        return ec;
      Archive::child_iterator member = memberOrErr.get();
      DEBUG_WITH_TYPE("FileArchive",
                      llvm::dbgs()
                          << llvm::format("0x%08llX ",
                                          (*member)->getBuffer()->data())
                          << "'" << name << "'\n");
      _symbolMemberMap.insert(std::make_pair(name, member));
    }
    return std::error_code();
  }

  typedef std::unordered_map<StringRef, Archive::child_iterator> MemberMap;
  typedef std::set<const char *> InstantiatedSet;

  std::shared_ptr<MemoryBuffer> _mb;
  const Registry &_registry;
  std::unique_ptr<Archive> _archive;
  MemberMap _symbolMemberMap;
  InstantiatedSet _membersInstantiated;
  bool _logLoading;
  std::vector<std::unique_ptr<MemoryBuffer>> _memberBuffers;
  std::vector<std::unique_ptr<File>> _filesReturned;
};

class ArchiveReader : public Reader {
public:
  ArchiveReader(bool logLoading) : _logLoading(logLoading) {}

  bool canParse(file_magic magic, MemoryBufferRef) const override {
    return magic == llvm::sys::fs::file_magic::archive;
  }

  ErrorOr<std::unique_ptr<File>> loadFile(std::unique_ptr<MemoryBuffer> mb,
                                          const Registry &reg) const override {
    StringRef path = mb->getBufferIdentifier();
    std::unique_ptr<File> ret =
        llvm::make_unique<FileArchive>(std::move(mb), reg, path, _logLoading);
    return std::move(ret);
  }

private:
  bool _logLoading;
};

} // anonymous namespace

void Registry::addSupportArchives(bool logLoading) {
  add(std::unique_ptr<Reader>(new ArchiveReader(logLoading)));
}

} // end namespace lld

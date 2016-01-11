//===- lib/ReaderWriter/ELF/ELFLinkingContext.cpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "ELFFile.h"
#include "OrderPass.h"
#include "TargetHandler.h"
#include "lld/Core/Instrumentation.h"
#include "lld/Core/SharedLibraryFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Config/config.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#if defined(HAVE_CXXABI_H)
#include <cxxabi.h>
#endif

using llvm::sys::fs::exists;
using llvm::sys::path::is_absolute;

namespace lld {

class CommandLineUndefinedAtom : public SimpleUndefinedAtom {
public:
  CommandLineUndefinedAtom(const File &f, StringRef name)
      : SimpleUndefinedAtom(f, name) {}

  CanBeNull canBeNull() const override {
    return CanBeNull::canBeNullAtBuildtime;
  }
};

void ELFLinkingContext::addPasses(PassManager &pm) {
  pm.add(llvm::make_unique<elf::OrderPass>());
}

uint16_t ELFLinkingContext::getOutputMachine() const {
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
    return llvm::ELF::EM_386;
  case llvm::Triple::x86_64:
    return llvm::ELF::EM_X86_64;
  case llvm::Triple::hexagon:
    return llvm::ELF::EM_HEXAGON;
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    return llvm::ELF::EM_MIPS;
  case llvm::Triple::aarch64:
    return llvm::ELF::EM_AARCH64;
  case llvm::Triple::arm:
    return llvm::ELF::EM_ARM;
  default:
    llvm_unreachable("Unhandled arch");
  }
}

StringRef ELFLinkingContext::entrySymbolName() const {
  if (_outputELFType == llvm::ELF::ET_EXEC && _entrySymbolName.empty())
    return "_start";
  return _entrySymbolName;
}

bool ELFLinkingContext::validateImpl(raw_ostream &diagnostics) {
  switch (outputFileType()) {
  case LinkingContext::OutputFileType::YAML:
    _writer = createWriterYAML(*this);
    break;
  default:
    _writer = createWriterELF(*this);
    break;
  }

  // If -dead_strip, set up initial live symbols.
  if (deadStrip())
    addDeadStripRoot(entrySymbolName());
  return true;
}

bool ELFLinkingContext::isDynamic() const {
  switch (_outputELFType) {
  case llvm::ELF::ET_EXEC:
    return !_isStaticExecutable;
  case llvm::ELF::ET_DYN:
    return true;
  }
  return false;
}

bool ELFLinkingContext::isRelativeReloc(const Reference &) const {
  return false;
}

Writer &ELFLinkingContext::writer() const { return *_writer; }

static void buildSearchPath(SmallString<128> &path, StringRef dir,
                            StringRef sysRoot) {
  if (dir.startswith("=/")) {
    // If a search directory begins with "=", "=" is replaced
    // with the sysroot path.
    path.assign(sysRoot);
    path.append(dir.substr(1));
  } else {
    path.assign(dir);
  }
}

ErrorOr<StringRef> ELFLinkingContext::searchLibrary(StringRef libName) const {
  bool hasColonPrefix = libName[0] == ':';
  SmallString<128> path;
  for (StringRef dir : _inputSearchPaths) {
    // Search for dynamic library
    if (!_isStaticExecutable) {
      buildSearchPath(path, dir, _sysrootPath);
      llvm::sys::path::append(path, hasColonPrefix
                                        ? libName.drop_front()
                                        : Twine("lib", libName) + ".so");
      if (exists(path.str()))
        return path.str().copy(_allocator);
    }
    // Search for static libraries too
    buildSearchPath(path, dir, _sysrootPath);
    llvm::sys::path::append(path, hasColonPrefix
                                      ? libName.drop_front()
                                      : Twine("lib", libName) + ".a");
    if (exists(path.str()))
      return path.str().copy(_allocator);
  }
  if (hasColonPrefix && exists(libName.drop_front()))
      return libName.drop_front();

  return make_error_code(llvm::errc::no_such_file_or_directory);
}

ErrorOr<StringRef> ELFLinkingContext::searchFile(StringRef fileName,
                                                 bool isSysRooted) const {
  SmallString<128> path;
  if (is_absolute(fileName) && isSysRooted) {
    path.assign(_sysrootPath);
    path.append(fileName);
    if (exists(path.str()))
      return path.str().copy(_allocator);
  } else if (exists(fileName)) {
    return fileName;
  }

  if (is_absolute(fileName))
    return make_error_code(llvm::errc::no_such_file_or_directory);

  for (StringRef dir : _inputSearchPaths) {
    buildSearchPath(path, dir, _sysrootPath);
    llvm::sys::path::append(path, fileName);
    if (exists(path.str()))
      return path.str().copy(_allocator);
  }
  return make_error_code(llvm::errc::no_such_file_or_directory);
}

void ELFLinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File>> &files) const {
  std::unique_ptr<SimpleFile> file(
      new SimpleFile("<internal file for --defsym>"));
  for (auto &i : getAbsoluteSymbols()) {
    StringRef sym = i.first;
    uint64_t val = i.second;
    file->addAtom(*(new (_allocator) SimpleAbsoluteAtom(
        *file, sym, Atom::scopeGlobal, val)));
  }
  files.push_back(std::move(file));
  LinkingContext::createInternalFiles(files);
}

void ELFLinkingContext::finalizeInputFiles() {
  // Add virtual archive that resolves undefined symbols.
  if (_resolver)
    getNodes().push_back(llvm::make_unique<FileNode>(std::move(_resolver)));
}

std::unique_ptr<File> ELFLinkingContext::createUndefinedSymbolFile() const {
  if (_initialUndefinedSymbols.empty())
    return nullptr;
  std::unique_ptr<SimpleFile> undefinedSymFile(
      new SimpleFile("command line option -u"));
  for (auto undefSymStr : _initialUndefinedSymbols)
    undefinedSymFile->addAtom(*(new (_allocator) CommandLineUndefinedAtom(
        *undefinedSymFile, undefSymStr)));
  return std::move(undefinedSymFile);
}

void ELFLinkingContext::notifySymbolTableCoalesce(const Atom *existingAtom,
                                                  const Atom *newAtom,
                                                  bool &useNew) {
  // First suppose that the `existingAtom` is defined
  // and the `newAtom` is undefined.
  auto *da = dyn_cast<DefinedAtom>(existingAtom);
  auto *ua = dyn_cast<UndefinedAtom>(newAtom);
  if (!da && !ua) {
    // Then try to reverse the assumption.
    da = dyn_cast<DefinedAtom>(newAtom);
    ua = dyn_cast<UndefinedAtom>(existingAtom);
  }

  if (da && ua && da->scope() == Atom::scopeGlobal &&
      isa<SharedLibraryFile>(ua->file()))
    // If strong defined atom coalesces away an atom declared
    // in the shared object the strong atom needs to be dynamically exported.
    // Save its name.
    _dynamicallyExportedSymbols.insert(ua->name());
}

std::string ELFLinkingContext::demangle(StringRef symbolName) const {
#if defined(HAVE_CXXABI_H)
  if (!demangleSymbols())
    return symbolName;

  // Only try to demangle symbols that look like C++ symbols
  if (!symbolName.startswith("_Z"))
    return symbolName;

  SmallString<256> symBuff;
  StringRef nullTermSym = Twine(symbolName).toNullTerminatedStringRef(symBuff);
  const char *cstr = nullTermSym.data();
  int status;
  char *demangled = abi::__cxa_demangle(cstr, nullptr, nullptr, &status);
  if (!demangled)
    return symbolName;
  std::string result(demangled);
  // __cxa_demangle() always uses a malloc'ed buffer to return the result.
  free(demangled);
  return result;
#else
  return symbolName;
#endif
}

void ELFLinkingContext::setUndefinesResolver(std::unique_ptr<File> resolver) {
  assert(isa<ArchiveLibraryFile>(resolver.get()) && "Wrong resolver type");
  _resolver = std::move(resolver);
}

void ELFLinkingContext::notifyInputSectionName(StringRef name) {
  // Save sections names which can be represented as a C identifier.
  if (name.find_first_not_of("0123456789"
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                             "abcdefghijklmnopqrstuvwxyz"
                             "_") == StringRef::npos) {
    std::lock_guard<std::mutex> lock(_cidentMutex);
    _cidentSections.insert(name);
  }
}

} // end namespace lld

//===--- ClangIndexRecordWriter.cpp - Index record serialization ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangIndexRecordWriter.h"
#include "FileIndexRecord.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexRecordReader.h"
#include "clang/Index/USRGeneration.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"

using namespace clang;
using namespace clang::index;

StringRef ClangIndexRecordWriter::getUSR(const Decl *D) {
  assert(D->isCanonicalDecl());
  auto Insert = USRByDecl.insert(std::make_pair(D, StringRef()));
  if (Insert.second) {
    Insert.first->second = getUSRNonCached(D);
  }
  return Insert.first->second;
}

StringRef ClangIndexRecordWriter::getUSR(const IdentifierInfo *Name,
                                         const MacroInfo *MI) {
  assert(Name && MI);
  auto Insert = USRByDecl.insert(std::make_pair(MI, StringRef()));
  if (Insert.second) {
    Insert.first->second = getUSRNonCached(Name, MI);
  }
  return Insert.first->second;
}

StringRef ClangIndexRecordWriter::getUSRNonCached(const Decl *D) {
  SmallString<256> Buf;
  bool Ignore = generateUSRForDecl(D, Buf);
  if (Ignore)
    return StringRef();
  StringRef USR = Buf.str();
  char *Ptr = Allocator.Allocate<char>(USR.size());
  std::copy(USR.begin(), USR.end(), Ptr);
  return StringRef(Ptr, USR.size());
}

StringRef ClangIndexRecordWriter::getUSRNonCached(const IdentifierInfo *Name,
                                                  const MacroInfo *MI) {
  SmallString<256> Buf;
  bool Ignore = generateUSRForMacro(Name->getName(), MI->getDefinitionLoc(),
                                    Ctx.getSourceManager(), Buf);
  if (Ignore)
    return StringRef();
  StringRef USR = Buf.str();
  char *Ptr = Allocator.Allocate<char>(USR.size());
  std::copy(USR.begin(), USR.end(), Ptr);
  return StringRef(Ptr, USR.size());
}

ClangIndexRecordWriter::ClangIndexRecordWriter(ASTContext &Ctx, bool Compress,
                                               RecordingOptions Opts)
    : Impl(Opts.DataDirPath, Compress), Ctx(Ctx), RecordOpts(std::move(Opts)) {
  if (Opts.RecordSymbolCodeGenName)
    ASTNameGen.reset(new ASTNameGenerator(Ctx));
}

ClangIndexRecordWriter::~ClangIndexRecordWriter() {}

bool ClangIndexRecordWriter::writeRecord(StringRef Filename,
                                         const FileIndexRecord &IdxRecord,
                                         std::string &Error,
                                         std::string *OutRecordFile) {

  std::array<uint8_t, 8> RecordHashArr = index::hashRecord(IdxRecord, Ctx);
  uint64_t RecordHash = 0;
  std::memcpy(&RecordHash, RecordHashArr.data(), RecordHashArr.size());

  switch (Impl.beginRecord(Filename, RecordHash, Error, OutRecordFile)) {
  case IndexRecordWriter::Result::Success:
    break; // Continue writing.
  case IndexRecordWriter::Result::Failure:
    return true;
  case IndexRecordWriter::Result::AlreadyExists:
    return false;
  }

  ASTContext &Ctx = getASTContext();
  SourceManager &SM = Ctx.getSourceManager();
  FileID FID = IdxRecord.getFileID();
  auto getLineCol = [&](unsigned Offset) -> std::pair<unsigned, unsigned> {
    unsigned LineNo = SM.getLineNumber(FID, Offset);
    unsigned ColNo = SM.getColumnNumber(FID, Offset);
    return std::make_pair(LineNo, ColNo);
  };

  llvm::DenseMap<const MacroInfo *, const IdentifierInfo *> MacroNames;

  for (auto &Occur : IdxRecord.getDeclOccurrencesSortedByOffset()) {
    unsigned Line, Col;
    std::tie(Line, Col) = getLineCol(Occur.Offset);
    SmallVector<writer::SymbolRelation, 3> Related;
    Related.reserve(Occur.Relations.size());
    for (auto &Rel : Occur.Relations)
      Related.push_back(writer::SymbolRelation{Rel.RelatedSymbol, Rel.Roles});
    if (Occur.MacroName)
      MacroNames[cast<const MacroInfo *>(Occur.DeclOrMacro)] = Occur.MacroName;

    Impl.addOccurrence(Occur.DeclOrMacro.getOpaqueValue(), Occur.Roles, Line,
                       Col, Related);
  }

  PrintingPolicy Policy(Ctx.getLangOpts());
  Policy.SuppressTemplateArgsInCXXConstructors = true;

  auto Result = Impl.endRecord(Error,
      [&](writer::OpaqueDecl OD, SmallVectorImpl<char> &Scratch) {
    writer::Symbol Sym;
    auto DeclOrMacro =
        llvm::PointerUnion<const Decl *, const MacroInfo *>::getFromOpaqueValue(
            const_cast<void *>(OD));
    if (auto *MI = DeclOrMacro.dyn_cast<const MacroInfo *>()) {
      auto *II = MacroNames[MI];
      assert(II);
      Sym.SymInfo = getSymbolInfoForMacro(*MI);
      Sym.Name = II->getName();
      Sym.USR = getUSR(II, MI);
      assert(!Sym.USR.empty() && "Recorded macro without USR!");
    } else {
      const Decl *D = cast<const Decl *>(DeclOrMacro);
      Sym.SymInfo = getSymbolInfo(D);

      auto *ND = dyn_cast<NamedDecl>(D);
      if (ND) {
        llvm::raw_svector_ostream OS(Scratch);
        DeclarationName DeclName = ND->getDeclName();
        if (!DeclName.isEmpty())
          DeclName.print(OS, Policy);
      }
      unsigned NameLen = Scratch.size();
      Sym.Name = StringRef(Scratch.data(), NameLen);

      Sym.USR = getUSR(D);
      assert(!Sym.USR.empty() && "Recorded decl without USR!");

      if (ASTNameGen && ND) {
        llvm::raw_svector_ostream OS(Scratch);
        ASTNameGen->writeName(ND, OS);
      }
      unsigned CGNameLen = Scratch.size() - NameLen;
      Sym.CodeGenName = StringRef(Scratch.data() + NameLen, CGNameLen);
    }

    return Sym;
  });

  switch (Result) {
  case IndexRecordWriter::Result::Success:
  case IndexRecordWriter::Result::AlreadyExists:
    return false;
  case IndexRecordWriter::Result::Failure:
    return true;
  }
}

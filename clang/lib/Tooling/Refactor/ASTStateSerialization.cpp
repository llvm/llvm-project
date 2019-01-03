//===-- ASTStateSerialization.cpp - Persists TU-specific state across TUs -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RefactoringContinuations.h"
#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::tooling::detail;

namespace {

class USRToDeclConverter
    : public clang::RecursiveASTVisitor<USRToDeclConverter> {
  llvm::StringMap<const Decl *> &USRs;
  unsigned NumFound = 0;

public:
  USRToDeclConverter(llvm::StringMap<const Decl *> &USRs) : USRs(USRs) {}

  bool isDone() const { return NumFound == USRs.size(); }

  bool VisitNamedDecl(const NamedDecl *D) {
    std::string USR = rename::getUSRForDecl(D);
    auto It = USRs.find(USR);
    if (It == USRs.end() || It->second)
      return true;
    It->second = D;
    ++NumFound;
    return NumFound != USRs.size();
  }
};

} // end anonymous namespace

const Decl *PersistentToASTSpecificStateConverter::lookupDecl(StringRef USR) {
  if (USR.empty())
    return nullptr;
  auto It = ConvertedDeclRefs.find(USR);
  if (It != ConvertedDeclRefs.end())
    return It->second;
  // FIXME: If we ever need to convert a PersistentDeclRef through the ASTQuery,
  // we have to support conversion without coalesced conversion.
  assert(false && "Persistent decl refs should be converted all at once");
  return nullptr;
}

void PersistentToASTSpecificStateConverter::runCoalescedConversions() {
  USRToDeclConverter Converter(ConvertedDeclRefs);
  for (Decl *D : Context.getTranslationUnitDecl()->decls()) {
    Converter.TraverseDecl(D);
    if (Converter.isDone())
      break;
  }
}

FileID
PersistentToASTSpecificStateConverter::convert(const PersistentFileID &Ref) {
  FileManager &FM = Context.getSourceManager().getFileManager();
  const FileEntry *Entry = FM.getFile(Ref.Filename);
  if (!Entry)
    return FileID();
  return Context.getSourceManager().translateFile(Entry);
}

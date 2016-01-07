//===- CIndexInclusionStack.cpp - Clang-C Source Indexing Library ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a callback mechanism for clients to get the inclusion
// stack from a translation unit.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXSourceLocation.h"
#include "CXTranslationUnit.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

static void getInclusions(const SrcMgr::SLocEntry &(SourceManager::*Getter)(unsigned, bool*) const, unsigned n,
                          CXTranslationUnit TU, CXInclusionVisitor CB,
                          CXClientData clientData)
{
  ASTUnit *CXXUnit = cxtu::getASTUnit(TU);
  SourceManager &SM = CXXUnit->getSourceManager();
  ASTContext &Ctx = CXXUnit->getASTContext();
  SmallVector<CXSourceLocation, 10> InclusionStack;
  const bool HasPreamble = SM.getPreambleFileID().isValid();

  for (unsigned i = 0 ; i < n ; ++i) {
    bool Invalid = false;
    const SrcMgr::SLocEntry &SL = (SM.*Getter)(i, &Invalid);

    if (!SL.isFile() || Invalid)
      continue;

    const SrcMgr::FileInfo &FI = SL.getFile();
    if (!FI.getContentCache()->OrigEntry)
      continue;

    // If this is the main file, and there is a preamble, skip this SLoc. The
    // inclusions of the preamble already showed it.
    SourceLocation L = FI.getIncludeLoc();
    if (HasPreamble && CXXUnit->isInMainFileID(L))
      continue;

    // Build the inclusion stack.
    InclusionStack.clear();
    while (L.isValid()) {
      PresumedLoc PLoc = SM.getPresumedLoc(L);
      InclusionStack.push_back(cxloc::translateSourceLocation(Ctx, L));
      L = PLoc.isValid()? PLoc.getIncludeLoc() : SourceLocation();
    }

    // If there is a preamble, the last entry is the "inclusion" of that
    // preamble into the main file, which has the bogus entry of main.c:1:1
    if (HasPreamble && !InclusionStack.empty())
      InclusionStack.pop_back();

    // Callback to the client.
    // FIXME: We should have a function to construct CXFiles.
    CB(static_cast<CXFile>(
         const_cast<FileEntry *>(FI.getContentCache()->OrigEntry)),
       InclusionStack.data(), InclusionStack.size(), clientData);
  }
}


extern "C" {
void clang_getInclusions(CXTranslationUnit TU, CXInclusionVisitor CB,
                         CXClientData clientData) {
  if (cxtu::isNotUsableTU(TU)) {
    LOG_BAD_TU(TU);
    return;
  }

  SourceManager &SM = cxtu::getASTUnit(TU)->getSourceManager();
  const unsigned n =  SM.local_sloc_entry_size();

  // In the case where all the SLocEntries are in an external source, traverse
  // those SLocEntries as well.  This is the case where we are looking
  // at the inclusion stack of an AST/PCH file. Also, if we are not looking at
  // a AST/PCH file, but this file has a pre-compiled preamble, we also need
  // to look in that file.
  if (n == 1 || SM.getPreambleFileID().isValid()) {
    getInclusions(&SourceManager::getLoadedSLocEntry,
                  SM.loaded_sloc_entry_size(), TU, CB, clientData);
  }

  // Not a PCH/AST file. Note, if there is a preamble, it could still be that
  // there are #includes in this file (e.g. for any include after the first
  // declaration).
  if (n != 1)
    getInclusions(&SourceManager::getLocalSLocEntry, n, TU, CB, clientData);

}
} // end extern C

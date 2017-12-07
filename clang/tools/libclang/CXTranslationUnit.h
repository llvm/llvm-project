//===- CXTranslationUnit.h - Routines for manipulating CXTranslationUnits -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXTranslationUnits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_LIBCLANG_CXTRANSLATIONUNIT_H
#define LLVM_CLANG_TOOLS_LIBCLANG_CXTRANSLATIONUNIT_H

#include "CLog.h"
#include "CXString.h"
#include "clang-c/Index.h"

namespace clang {
  class ASTUnit;
  class CIndexer;
namespace index {
class CommentToXMLConverter;
} // namespace index
} // namespace clang

struct CXTranslationUnitImpl {
  clang::CIndexer *CIdx;
  clang::ASTUnit *TheASTUnit;
  clang::cxstring::CXStringPool *StringPool;
  void *Diagnostics;
  void *OverridenCursorsPool;
  clang::index::CommentToXMLConverter *CommentToXML;
  unsigned ParsingOptions;
  std::vector<std::string> Arguments;
};

struct CXTargetInfoImpl {
  CXTranslationUnit TranslationUnit;
};

namespace clang {
namespace cxtu {

CXTranslationUnitImpl *MakeCXTranslationUnit(CIndexer *CIdx,
                                             std::unique_ptr<ASTUnit> AU);

static inline ASTUnit *getASTUnit(CXTranslationUnit TU) {
  if (!TU)
    return nullptr;
  return TU->TheASTUnit;
}

/// \returns true if the ASTUnit has a diagnostic about the AST file being
/// corrupted.
bool isASTReadError(ASTUnit *AU);

static inline bool isNotUsableTU(CXTranslationUnit TU) {
  return !TU;
}

#define LOG_BAD_TU(TU)                                  \
    do {                                                \
      LOG_FUNC_SECTION {                                \
        *Log << "called with a bad TU: " << TU;         \
      }                                                 \
    } while(false)

class CXTUOwner {
  CXTranslationUnitImpl *TU;
  
public:
  CXTUOwner(CXTranslationUnitImpl *tu) : TU(tu) { }
  ~CXTUOwner();

  CXTranslationUnitImpl *getTU() const { return TU; }

  CXTranslationUnitImpl *takeTU() {
    CXTranslationUnitImpl *retTU = TU;
    TU = nullptr;
    return retTU;
  }
};


}} // end namespace clang::cxtu

#endif

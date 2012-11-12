/*===-- CIndexDiagnostic.h - Diagnostics C Interface ------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Implements the diagnostic functions of the Clang C interface.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
#ifndef LLVM_CLANG_CINDEX_DIAGNOSTIC_H
#define LLVM_CLANG_CINDEX_DIAGNOSTIC_H

#include "clang-c/Index.h"
#include <vector>
#include <assert.h>

namespace clang {

class LangOptions;
class StoredDiagnostic;
class CXDiagnosticImpl;
  
class CXDiagnosticSetImpl {
  std::vector<CXDiagnosticImpl *> Diagnostics;
  const bool IsExternallyManaged;
public:
  CXDiagnosticSetImpl(bool isManaged = false)
    : IsExternallyManaged(isManaged) {}

  virtual ~CXDiagnosticSetImpl();
  
  size_t getNumDiagnostics() const {
    return Diagnostics.size();
  }
  
  CXDiagnosticImpl *getDiagnostic(unsigned i) const {
    assert(i < getNumDiagnostics());
    return Diagnostics[i];
  }
  
  void appendDiagnostic(CXDiagnosticImpl *D) {
    Diagnostics.push_back(D);
  }
  
  bool empty() const {
    return Diagnostics.empty();
  }
  
  bool isExternallyManaged() const { return IsExternallyManaged; }
};

class CXDiagnosticImpl {
public:
  enum Kind { StoredDiagnosticKind, LoadedDiagnosticKind,
              CustomNoteDiagnosticKind };
  
  virtual ~CXDiagnosticImpl();
  
  /// \brief Return the severity of the diagnostic.
  virtual CXDiagnosticSeverity getSeverity() const = 0;
  
  /// \brief Return the location of the diagnostic.
  virtual CXSourceLocation getLocation() const = 0;

  /// \brief Return the spelling of the diagnostic.
  virtual CXString getSpelling() const = 0;

  /// \brief Return the text for the diagnostic option.
  virtual CXString getDiagnosticOption(CXString *Disable) const = 0;
  
  /// \brief Return the category of the diagnostic.
  virtual unsigned getCategory() const = 0;

  /// \brief Return the category string of the diagnostic.
  virtual CXString getCategoryText() const = 0;

  /// \brief Return the number of source ranges for the diagnostic.
  virtual unsigned getNumRanges() const = 0;
  
  /// \brief Return the source ranges for the diagnostic.
  virtual CXSourceRange getRange(unsigned Range) const = 0;

  /// \brief Return the number of FixIts.
  virtual unsigned getNumFixIts() const = 0;

  /// \brief Return the FixIt information (source range and inserted text).
  virtual CXString getFixIt(unsigned FixIt,
                            CXSourceRange *ReplacementRange) const = 0;

  Kind getKind() const { return K; }
  
  CXDiagnosticSetImpl &getChildDiagnostics() {
    return ChildDiags;
  }
  
protected:
  CXDiagnosticImpl(Kind k) : K(k) {}
  CXDiagnosticSetImpl ChildDiags;
  
  void append(CXDiagnosticImpl *D) {
    ChildDiags.appendDiagnostic(D);
  }
  
private:
  Kind K;
};
  
/// \brief The storage behind a CXDiagnostic
struct CXStoredDiagnostic : public CXDiagnosticImpl {
  const StoredDiagnostic &Diag;
  const LangOptions &LangOpts;
  
  CXStoredDiagnostic(const StoredDiagnostic &Diag,
                     const LangOptions &LangOpts)
    : CXDiagnosticImpl(StoredDiagnosticKind),
      Diag(Diag), LangOpts(LangOpts) { }
  
  virtual ~CXStoredDiagnostic() {}
  
  /// \brief Return the severity of the diagnostic.
  virtual CXDiagnosticSeverity getSeverity() const;
  
  /// \brief Return the location of the diagnostic.
  virtual CXSourceLocation getLocation() const;

  /// \brief Return the spelling of the diagnostic.
  virtual CXString getSpelling() const;

  /// \brief Return the text for the diagnostic option.
  virtual CXString getDiagnosticOption(CXString *Disable) const;
  
  /// \brief Return the category of the diagnostic.
  virtual unsigned getCategory() const;
  
  /// \brief Return the category string of the diagnostic.
  virtual CXString getCategoryText() const;

  /// \brief Return the number of source ranges for the diagnostic.
  virtual unsigned getNumRanges() const;
  
  /// \brief Return the source ranges for the diagnostic.
  virtual CXSourceRange getRange(unsigned Range) const;

  /// \brief Return the number of FixIts.
  virtual unsigned getNumFixIts() const;

  /// \brief Return the FixIt information (source range and inserted text).
  virtual CXString getFixIt(unsigned FixIt,
                            CXSourceRange *ReplacementRange) const;

  static bool classof(const CXDiagnosticImpl *D) {
    return D->getKind() == StoredDiagnosticKind;
  }
};

namespace cxdiag {
CXDiagnosticSetImpl *lazyCreateDiags(CXTranslationUnit TU,
                                     bool checkIfChanged = false);
} // end namespace cxdiag

} // end namespace clang

#endif // LLVM_CLANG_CINDEX_DIAGNOSTIC_H

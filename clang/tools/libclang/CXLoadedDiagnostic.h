/*===-- CXLoadedDiagnostic.h - Handling of persisent diags ------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Implements handling of persisent diagnostics.                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_CLANG_CINDEX_LOADED_DIAGNOSTIC_H
#define LLVM_CLANG_CINDEX_LOADED_DIAGNOSTIC_H

#include "CIndexDiagnostic.h"
#include "llvm/ADT/StringRef.h"
#include "clang/Basic/LLVM.h"
#include <string>
#include <vector>

namespace clang {
class CXLoadedDiagnostic : public CXDiagnosticImpl {
public:
  CXLoadedDiagnostic() : CXDiagnosticImpl(LoadedDiagnosticKind),
    severity(0), category(0) {}

  virtual ~CXLoadedDiagnostic();
  
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
    return D->getKind() == LoadedDiagnosticKind;
  }
  
  /// \brief Decode the CXSourceLocation into file, line, column, and offset.
  static void decodeLocation(CXSourceLocation location,
                             CXFile *file,
                             unsigned *line,
                             unsigned *column,
                             unsigned *offset);

  struct Location {
    CXFile file;
    unsigned line;
    unsigned column;
    unsigned offset;
    
    Location() : line(0), column(0), offset(0) {}    
  };
  
  Location DiagLoc;

  std::vector<CXSourceRange> Ranges;
  std::vector<std::pair<CXSourceRange, CXString> > FixIts;
  llvm::StringRef Spelling;
  llvm::StringRef DiagOption;
  llvm::StringRef CategoryText;
  unsigned severity;
  unsigned category;
};
}

#endif

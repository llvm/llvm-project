//===- ExternalPreprocessorSource.h - Abstract Macro Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ExternalPreprocessorSource interface, which enables
//  construction of macro definitions from some external source.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LEX_EXTERNAL_PREPROCESSOR_SOURCE_H
#define LLVM_CLANG_LEX_EXTERNAL_PREPROCESSOR_SOURCE_H

namespace clang {
  
/// \brief Abstract interface for external sources of preprocessor 
/// information.
///
/// This abstract class allows an external sources (such as the \c ASTReader) 
/// to provide additional macro definitions.
class ExternalPreprocessorSource {
public:
  virtual ~ExternalPreprocessorSource();
  
  /// \brief Read the set of macros defined by this external macro source.
  virtual void ReadDefinedMacros() = 0;
  
  /// \brief Update an out-of-date identifier.
  virtual void updateOutOfDateIdentifier(IdentifierInfo &II) = 0;
};
  
}

#endif // LLVM_CLANG_LEX_EXTERNAL_PREPROCESSOR_SOURCE_H

//== SMTContext.h -----------------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SMT generic Context API, which will be the base class
//  for every SMT solver context specific class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTCONTEXT_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTCONTEXT_H

namespace clang {
namespace ento {

/// Generic base class for SMT contexts
class SMTContext {
public:
  SMTContext() = default;
  virtual ~SMTContext() = default;
};

} // namespace ento
} // namespace clang

#endif

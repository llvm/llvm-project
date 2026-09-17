//===- PrettyStackTraceStackFrame.h - show analysis backtrace --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CORE_PRETTYSTACKTRACELOCATIONCONTEXT_H
#define LLVM_CLANG_LIB_STATICANALYZER_CORE_PRETTYSTACKTRACELOCATIONCONTEXT_H

#include "clang/Analysis/AnalysisDeclContext.h"

namespace clang {
namespace ento {

/// While alive, includes the current analysis stack in a crash trace.
///
/// Example:
/// \code
/// 0.     Program arguments: ...
/// 1.     <eof> parser at end of file
/// 2.     While analyzing stack:
///        #0 void inlined()
///        #1 void test()
/// 3.     crash-trace.c:6:3: Error evaluating statement
/// \endcode
class PrettyStackTraceStackFrame : public llvm::PrettyStackTraceEntry {
  const StackFrame *SF;

public:
  PrettyStackTraceStackFrame(const StackFrame *SF) : SF(SF) { assert(SF); }

  void print(raw_ostream &Out) const override {
    Out << "While analyzing stack: \n";
    SF->dumpStack(Out);
  }
};

} // namespace ento
} // namespace clang

#endif

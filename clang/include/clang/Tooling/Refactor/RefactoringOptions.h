//===--- RefactoringOptions.h - A set of all the refactoring options ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a set of all possible refactoring options that can be
// given to the refactoring operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPTIONS_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPTIONS_H

#include "clang/AST/DeclBase.h"
#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Refactor/RefactoringOptionSet.h"

namespace clang {
namespace tooling {
namespace option {

namespace detail {

struct BoolOptionBase : RefactoringOption {
protected:
  bool Value = false;
  void serializeImpl(const SerializationContext &Context, const char *Name);

public:
  operator bool() const { return Value; }
};

template <typename Option> struct BoolOption : BoolOptionBase {
  void serialize(const SerializationContext &Context) override {
    serializeImpl(Context, Option::Name);
  }

  static Option getTrue() {
    Option Result;
    Result.Value = true;
    return Result;
  }
};

} // end namespace detail

struct AvoidTextualMatches final : detail::BoolOption<AvoidTextualMatches> {
  static constexpr const char *Name = "rename.avoid.textual.matches";
};

} // end namespace option
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPTIONS_H

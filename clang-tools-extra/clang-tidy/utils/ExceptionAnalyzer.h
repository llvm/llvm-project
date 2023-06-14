//===--- ExceptionAnalyzer.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_ANALYZER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_ANALYZER_H

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"

namespace clang::tidy::utils {

/// This class analysis if a `FunctionDecl` can in principle throw an
/// exception, either directly or indirectly. It can be configured to ignore
/// custom exception types.
class ExceptionAnalyzer {
public:
  enum class State {
    Throwing,    ///< The function can definitely throw given an AST.
    NotThrowing, ///< This function can not throw, given an AST.
    Unknown,     ///< This can happen for extern functions without available
                 ///< definition.
  };

  /// Bundle the gathered information about an entity like a function regarding
  /// it's exception behaviour. The 'NonThrowing'-state can be considered as the
  /// neutral element in terms of information propagation.
  /// In the case of 'Throwing' state it is possible that 'getExceptionTypes'
  /// does not include *ALL* possible types as there is the possibility that
  /// an 'Unknown' function is called that might throw a previously unknown
  /// exception at runtime.
  class ExceptionInfo {
  public:
    using Throwables = llvm::SmallSet<const Type *, 2>;
    static ExceptionInfo createUnknown() {
      return ExceptionInfo(State::Unknown);
    }
    static ExceptionInfo createNonThrowing() {
      return ExceptionInfo(State::Throwing);
    }

    /// By default the exception situation is unknown and must be
    /// clarified step-wise.
    ExceptionInfo() : Behaviour(State::NotThrowing), ContainsUnknown(false) {}
    ExceptionInfo(State S)
        : Behaviour(S), ContainsUnknown(S == State::Unknown) {}

    ExceptionInfo(const ExceptionInfo &) = default;
    ExceptionInfo &operator=(const ExceptionInfo &) = default;
    ExceptionInfo(ExceptionInfo &&) = default;
    ExceptionInfo &operator=(ExceptionInfo &&) = default;

    State getBehaviour() const { return Behaviour; }

    /// Register a single exception type as recognized potential exception to be
    /// thrown.
    void registerException(const Type *ExceptionType);

    /// Registers a `SmallVector` of exception types as recognized potential
    /// exceptions to be thrown.
    void registerExceptions(const Throwables &Exceptions);

    /// Updates the local state according to the other state. That means if
    /// for example a function contains multiple statements the 'ExceptionInfo'
    /// for the final function is the merged result of each statement.
    /// If one of these statements throws the whole function throws and if one
    /// part is unknown and the rest is non-throwing the result will be
    /// unknown.
    ExceptionInfo &merge(const ExceptionInfo &Other);

    /// This method is useful in case 'catch' clauses are analyzed as it is
    /// possible to catch multiple exception types by one 'catch' if they
    /// are a subclass of the 'catch'ed exception type.
    /// Returns 'true' if some exceptions were filtered, otherwise 'false'.
    bool filterByCatch(const Type *BaseClass, const ASTContext &Context);

    /// Filter the set of thrown exception type against a set of ignored
    /// types that shall not be considered in the exception analysis.
    /// This includes explicit `std::bad_alloc` ignoring as separate option.
    ExceptionInfo &
    filterIgnoredExceptions(const llvm::StringSet<> &IgnoredTypes,
                            bool IgnoreBadAlloc);

    /// Clear the state to 'NonThrowing' to make the corresponding entity
    /// neutral.
    void clear();

    /// References the set of known exception types that can escape from the
    /// corresponding entity.
    const Throwables &getExceptionTypes() const { return ThrownExceptions; }

    /// Signal if the there is any 'Unknown' element within the scope of
    /// the related entity. This might be relevant if the entity is 'Throwing'
    /// and to ensure that no other exception then 'getExceptionTypes' can
    /// occur. If there is an 'Unknown' element this can not be guaranteed.
    bool containsUnknownElements() const { return ContainsUnknown; }

  private:
    /// Recalculate the 'Behaviour' for example after filtering.
    void reevaluateBehaviour();

    /// Keep track if the entity related to this 'ExceptionInfo' can in princple
    /// throw, if it's unknown or if it won't throw.
    State Behaviour;

    /// Keep track if the entity contains any unknown elements to keep track
    /// of the certainty of decisions and/or correct 'Behaviour' transition
    /// after filtering.
    bool ContainsUnknown;

    /// 'ThrownException' is empty if the 'Behaviour' is either 'NotThrowing' or
    /// 'Unknown'.
    Throwables ThrownExceptions;
  };

  ExceptionAnalyzer() = default;

  void ignoreBadAlloc(bool ShallIgnore) { IgnoreBadAlloc = ShallIgnore; }
  void ignoreExceptions(llvm::StringSet<> ExceptionNames) {
    IgnoredExceptions = std::move(ExceptionNames);
  }

  ExceptionInfo analyze(const FunctionDecl *Func);
  ExceptionInfo analyze(const Stmt *Stmt);

private:
  ExceptionInfo
  throwsException(const FunctionDecl *Func,
                  const ExceptionInfo::Throwables &Caught,
                  llvm::SmallSet<const FunctionDecl *, 32> &CallStack);
  ExceptionInfo
  throwsException(const Stmt *St, const ExceptionInfo::Throwables &Caught,
                  llvm::SmallSet<const FunctionDecl *, 32> &CallStack);

  ExceptionInfo analyzeImpl(const FunctionDecl *Func);
  ExceptionInfo analyzeImpl(const Stmt *Stmt);

  template <typename T> ExceptionInfo analyzeDispatch(const T *Node);

  bool IgnoreBadAlloc = true;
  llvm::StringSet<> IgnoredExceptions;
  llvm::DenseMap<const FunctionDecl *, ExceptionInfo> FunctionCache{32u};
};

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXCEPTION_ANALYZER_H

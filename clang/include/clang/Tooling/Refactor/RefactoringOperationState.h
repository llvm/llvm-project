//===--- RefactoringOperationState.h - Serializable operation state -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the refactoring operation state types that represent the
// TU-independent state that is used for refactoring continuations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPERATION_STATE_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPERATION_STATE_H

#include "clang/AST/Decl.h"
#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Refactor/USRFinder.h"
#include <string>
#include <type_traits>

namespace clang {
namespace tooling {

namespace detail {

struct PersistentDeclRefBase {};

} // end namespace detail

/// Declaration references are persisted across translation units by using
/// USRs.
template <typename T>
struct PersistentDeclRef : std::enable_if<std::is_base_of<Decl, T>::value,
                                          detail::PersistentDeclRefBase>::type {
  std::string USR;
  // FIXME: We can improve the efficiency of conversion to Decl * by storing the
  // decl kind.

  PersistentDeclRef(std::string USR) : USR(std::move(USR)) {}
  PersistentDeclRef(PersistentDeclRef &&Other) = default;
  PersistentDeclRef &operator=(PersistentDeclRef &&Other) = default;
  PersistentDeclRef(const PersistentDeclRef &Other) = default;
  PersistentDeclRef &operator=(const PersistentDeclRef &Other) = default;

  static PersistentDeclRef<T> create(const Decl *D) {
    // FIXME: Move the getUSRForDecl method somewhere else.
    return PersistentDeclRef<T>(rename::getUSRForDecl(D));
  }
};

/// FileIDs are persisted across translation units by using filenames.
struct PersistentFileID {
  std::string Filename;

  PersistentFileID(std::string Filename) : Filename(std::move(Filename)) {}
  PersistentFileID(PersistentFileID &&Other) = default;
  PersistentFileID &operator=(PersistentFileID &&Other) = default;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPERATION_STATE_H

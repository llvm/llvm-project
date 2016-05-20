//===--- MakeSharedCheck.h - clang-tidy--------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SHARED_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SHARED_H

#include "MakeSmartPtrCheck.h"

namespace clang {
namespace tidy {
namespace modernize {

/// Replace the pattern:
/// \code
///   std::shared_ptr<type>(new type(args...))
/// \endcode
///
/// With the safer version:
/// \code
///   std::make_shared<type>(args...)
/// \endcode
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize-make-shared.html
class MakeSharedCheck : public MakeSmartPtrCheck {
public:
  MakeSharedCheck(StringRef Name, ClangTidyContext *Context);

protected:
  SmartPtrTypeMatcher getSmartPointerTypeMatcher() const override;
};

} // namespace modernize
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_MAKE_SHARED_H

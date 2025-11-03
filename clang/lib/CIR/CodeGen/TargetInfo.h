//===---- TargetInfo.h - Encapsulate target details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function definition used
// to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_TARGETINFO_H
#define LLVM_CLANG_LIB_CIR_TARGETINFO_H

#include "ABIInfo.h"
#include "CIRGenTypes.h"
#include "clang/Basic/AddressSpaces.h"

#include <memory>
#include <utility>

namespace clang::CIRGen {

/// isEmptyFieldForLayout - Return true if the field is "empty", that is,
/// either a zero-width bit-field or an isEmptyRecordForLayout.
bool isEmptyFieldForLayout(const ASTContext &context, const FieldDecl *fd);

/// isEmptyRecordForLayout - Return true if a structure contains only empty
/// base classes (per  isEmptyRecordForLayout) and fields (per
/// isEmptyFieldForLayout). Note, C++ record fields are considered empty
/// if the [[no_unique_address]] attribute would have made them empty.
bool isEmptyRecordForLayout(const ASTContext &context, QualType t);

class TargetCIRGenInfo {
  std::unique_ptr<ABIInfo> info;

public:
  TargetCIRGenInfo(std::unique_ptr<ABIInfo> info) : info(std::move(info)) {}

  virtual ~TargetCIRGenInfo() = default;

  /// Returns ABI info helper for the target.
  const ABIInfo &getABIInfo() const { return *info; }

  /// Get the address space for alloca.
  virtual cir::TargetAddressSpaceAttr getCIRAllocaAddressSpace() const {
    return {};
  }

  /// Determine whether a call to an unprototyped functions under
  /// the given calling convention should use the variadic
  /// convention or the non-variadic convention.
  ///
  /// There's a good reason to make a platform's variadic calling
  /// convention be different from its non-variadic calling
  /// convention: the non-variadic arguments can be passed in
  /// registers (better for performance), and the variadic arguments
  /// can be passed on the stack (also better for performance).  If
  /// this is done, however, unprototyped functions *must* use the
  /// non-variadic convention, because C99 states that a call
  /// through an unprototyped function type must succeed if the
  /// function was defined with a non-variadic prototype with
  /// compatible parameters.  Therefore, splitting the conventions
  /// makes it impossible to call a variadic function through an
  /// unprototyped type.  Since function prototypes came out in the
  /// late 1970s, this is probably an acceptable trade-off.
  /// Nonetheless, not all platforms are willing to make it, and in
  /// particularly x86-64 bends over backwards to make the
  /// conventions compatible.
  ///
  /// The default is false.  This is correct whenever:
  ///   - the conventions are exactly the same, because it does not
  ///     matter and the resulting IR will be somewhat prettier in
  ///     certain cases; or
  ///   - the conventions are substantively different in how they pass
  ///     arguments, because in this case using the variadic convention
  ///     will lead to C99 violations.
  ///
  /// However, some platforms make the conventions identical except
  /// for passing additional out-of-band information to a variadic
  /// function: for example, x86-64 passes the number of SSE
  /// arguments in %al.  On these platforms, it is desirable to
  /// call unprototyped functions using the variadic convention so
  /// that unprototyped calls to varargs functions still succeed.
  ///
  /// Relatedly, platforms which pass the fixed arguments to this:
  ///   A foo(B, C, D);
  /// differently than they would pass them to this:
  ///   A foo(B, C, D, ...);
  /// may need to adjust the debugger-support code in Sema to do the
  /// right thing when calling a function with no know signature.
  virtual bool isNoProtoCallVariadic(const FunctionNoProtoType *fnType) const;
};

std::unique_ptr<TargetCIRGenInfo> createX8664TargetCIRGenInfo(CIRGenTypes &cgt);

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_TARGETINFO_H

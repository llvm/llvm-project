//===-- include/flang/Evaluate/intrinsics-library.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_
#define FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_

// Defines structures to be used in F18 for folding intrinsic function with host
// runtime libraries.

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace Fortran::evaluate {
class FoldingContext;
class DynamicType;
struct SomeType;
template <typename> class Expr;

// Define a callable type that is used to fold scalar intrinsic function using
// host runtime. These callables are responsible for the conversions between
// host types and Fortran abstract types (Scalar<T>). They also deal with
// floating point environment (To set it up to match the Fortran compiling
// options and to clean it up after the call). Floating point errors are
// reported to the FoldingContext. For 16bits float types, 32bits float host
// runtime plus conversions may be used to build the host wrappers if no 16bits
// runtime is available. IEEE 128bits float may also be used for x87 float.
// Potential conversion overflows are reported by the HostRuntimeWrapper in the
// FoldingContext.
using HostRuntimeWrapper = std::function<Expr<SomeType>(
    FoldingContext &, std::vector<Expr<SomeType>> &&)>;

// Returns the folder using host runtime given the intrinsic function name,
// result and argument types. Nullopt if no host runtime is available for such
// intrinsic function.
std::optional<HostRuntimeWrapper> GetHostRuntimeWrapper(const std::string &name,
    DynamicType resultType, const std::vector<DynamicType> &argTypes);
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_

//===-- Support.h - generate support runtime API calls ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_SUPPORT_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_SUPPORT_H

namespace aiir {
class Value;
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate call to `CopyAndUpdateDescriptor` runtime routine.
void genCopyAndUpdateDescriptor(fir::FirOpBuilder &builder, aiir::Location loc,
                                aiir::Value to, aiir::Value from,
                                aiir::Value newDynamicType,
                                aiir::Value newAttribute,
                                aiir::Value newLowerBounds);

/// Generate call to `IsAssumedSize` runtime routine.
aiir::Value genIsAssumedSize(fir::FirOpBuilder &builder, aiir::Location loc,
                             aiir::Value box);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_SUPPORT_H

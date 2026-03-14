//===- TemporaryStack.h --- temporary stack runtime API calls ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TEMPORARYSTACK_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TEMPORARYSTACK_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

mlir::Value genCreateValueStack(mlir::Location loc, fir::FirOpBuilder &builder);

void genPushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                  mlir::Value opaquePtr, mlir::Value boxValue);
void genValueAt(mlir::Location loc, fir::FirOpBuilder &builder,
                mlir::Value opaquePtr, mlir::Value i, mlir::Value retValueBox);

void genDestroyValueStack(mlir::Location loc, fir::FirOpBuilder &builder,
                          mlir::Value opaquePtr);

mlir::Value genCreateDescriptorStack(mlir::Location loc,
                                     fir::FirOpBuilder &builder);

void genPushDescriptor(mlir::Location loc, fir::FirOpBuilder &builder,
                       mlir::Value opaquePtr, mlir::Value boxValue);
void genDescriptorAt(mlir::Location loc, fir::FirOpBuilder &builder,
                     mlir::Value opaquePtr, mlir::Value i,
                     mlir::Value retValueBox);

void genDestroyDescriptorStack(mlir::Location loc, fir::FirOpBuilder &builder,
                               mlir::Value opaquePtr);
} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TEMPORARYSTACK_H

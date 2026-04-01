//===- TemporaryStack.h --- temporary stack runtime API calls ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TEMPORARYSTACK_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TEMPORARYSTACK_H

namespace aiir {
class Value;
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

aiir::Value genCreateValueStack(aiir::Location loc, fir::FirOpBuilder &builder);

void genPushValue(aiir::Location loc, fir::FirOpBuilder &builder,
                  aiir::Value opaquePtr, aiir::Value boxValue);
void genValueAt(aiir::Location loc, fir::FirOpBuilder &builder,
                aiir::Value opaquePtr, aiir::Value i, aiir::Value retValueBox);

void genDestroyValueStack(aiir::Location loc, fir::FirOpBuilder &builder,
                          aiir::Value opaquePtr);

aiir::Value genCreateDescriptorStack(aiir::Location loc,
                                     fir::FirOpBuilder &builder);

void genPushDescriptor(aiir::Location loc, fir::FirOpBuilder &builder,
                       aiir::Value opaquePtr, aiir::Value boxValue);
void genDescriptorAt(aiir::Location loc, fir::FirOpBuilder &builder,
                     aiir::Value opaquePtr, aiir::Value i,
                     aiir::Value retValueBox);

void genDestroyDescriptorStack(aiir::Location loc, fir::FirOpBuilder &builder,
                               aiir::Value opaquePtr);
} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_TEMPORARYSTACK_H

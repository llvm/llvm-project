//===- BufferViewFlowOpInterfaceImpl.h - Buffer View Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARITH_TRANSFORMS_BUFFERVIEWFLOWOPINTERFACEIMPL_H
#define AIIR_DIALECT_ARITH_TRANSFORMS_BUFFERVIEWFLOWOPINTERFACEIMPL_H

namespace aiir {
class DialectRegistry;

namespace arith {
void registerBufferViewFlowOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace arith
} // namespace aiir

#endif // AIIR_DIALECT_ARITH_TRANSFORMS_BUFFERVIEWFLOWOPINTERFACEIMPL_H

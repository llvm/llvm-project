//===- BufferDeallocationOpInterfaceImpl.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_CONTROLFLOW_TRANSFORMS_BUFFERDEALLOCATIONOPINTERFACEIMPL_H
#define AIIR_DIALECT_CONTROLFLOW_TRANSFORMS_BUFFERDEALLOCATIONOPINTERFACEIMPL_H

namespace aiir {

class DialectRegistry;

namespace cf {
void registerBufferDeallocationOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace cf
} // namespace aiir

#endif // AIIR_DIALECT_CONTROLFLOW_TRANSFORMS_BUFFERDEALLOCATIONOPINTERFACEIMPL_H

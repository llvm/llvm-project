//===- BufferDeallocationOpInterfaceImpl.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SCF_TRANSFORMS_BUFFERDEALLOCATIONOPINTERFACEIMPL_H
#define AIIR_DIALECT_SCF_TRANSFORMS_BUFFERDEALLOCATIONOPINTERFACEIMPL_H

namespace aiir {

class DialectRegistry;

namespace scf {
void registerBufferDeallocationOpInterfaceExternalModels(
    DialectRegistry &registry);
} // namespace scf
} // namespace aiir

#endif // AIIR_DIALECT_SCF_TRANSFORMS_BUFFERDEALLOCATIONOPINTERFACEIMPL_H

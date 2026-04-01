//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SHAPE_BUFFERIZABLEOPINTERFACEIMPL_H
#define AIIR_DIALECT_SHAPE_BUFFERIZABLEOPINTERFACEIMPL_H

namespace aiir {
class DialectRegistry;

namespace shape {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace shape
} // namespace aiir

#endif // AIIR_DIALECT_SHAPE_BUFFERIZABLEOPINTERFACEIMPL_H

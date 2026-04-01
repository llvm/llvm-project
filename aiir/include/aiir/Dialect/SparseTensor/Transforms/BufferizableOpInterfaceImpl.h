//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SPARSETENSOR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define AIIR_DIALECT_SPARSETENSOR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace aiir {
class DialectRegistry;

namespace sparse_tensor {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace sparse_tensor
} // namespace aiir

#endif // AIIR_DIALECT_SPARSETENSOR_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

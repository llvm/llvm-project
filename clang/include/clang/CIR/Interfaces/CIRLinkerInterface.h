//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INTERFACES_CIR_CIRLINKINTERFACE_H_
#define CLANG_INTERFACES_CIR_CIRLINKINTERFACE_H_

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace cir {
void registerLinkerInterface(mlir::DialectRegistry &registry);
} // namespace cir

#endif // CLANG_INTERFACES_CIR_CIRLINKINTERFACE_H_

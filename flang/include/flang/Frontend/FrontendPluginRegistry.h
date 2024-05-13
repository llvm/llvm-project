//===- FrontendPluginRegistry.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pluggable Frontend Action Interface
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_FRONTEND_FRONTENDPLUGINREGISTRY_H
#define FORTRAN_FRONTEND_FRONTENDPLUGINREGISTRY_H

#include "flang/Frontend/FrontendActions.h"
#include "llvm/Support/Registry.h"

namespace Fortran::frontend {

/// The frontend plugin registry.
using FrontendPluginRegistry = llvm::Registry<PluginParseTreeAction>;

} // namespace Fortran::frontend

#endif // FORTRAN_FRONTEND_FRONTENDPLUGINREGISTRY_H

//===-- EnvironmentDefaults.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// EnvironmentDefaults is a list of default values for environment variables
// that may be specified at compile time and set by the runtime during
// program startup if the variable is not already present in the environment.
// EnvironmentDefaults is intended to allow options controlled by environment
// variables to also be set on the command line at compile time without needing
// to define option-specific runtime calls or duplicate logic within the
// runtime. For example, the -fconvert command line option is implemented in
// terms of an default value for the FORT_CONVERT environment variable.

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ENVIRONMENTDEFAULTS_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ENVIRONMENTDEFAULTS_H

#include <vector>

namespace fir {
class FirOpBuilder;
class GlobalOp;
} // namespace fir

namespace mlir {
class Location;
class Value;
} // namespace mlir

namespace Fortran::lower {
struct EnvironmentDefault;
} // namespace Fortran::lower

namespace fir::runtime {

/// Create the list of environment variable defaults for the runtime to set. The
/// form of the generated list is defined in the runtime header file
/// environment-default-list.h
mlir::Value genEnvironmentDefaults(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const std::vector<Fortran::lower::EnvironmentDefault> &envDefaults);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ENVIRONMENTDEFAULTS_H

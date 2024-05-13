//===-- Lower/EnvironmentDefault.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_ENVIRONMENTDEFAULT_H
#define FORTRAN_LOWER_ENVIRONMENTDEFAULT_H

#include <string>

namespace Fortran::lower {

struct EnvironmentDefault {
  std::string varName;
  std::string defaultValue;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_ENVIRONMENTDEFAULT_H

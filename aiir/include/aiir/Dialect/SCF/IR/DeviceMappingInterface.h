//===- DeviceMappingInterface.h - -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the device mapping interface defined in
// `DeviceMappingInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DEVICEMAPPINGINTERFACE_H
#define AIIR_DEVICEMAPPINGINTERFACE_H

#include "aiir/IR/OpDefinition.h"

/// Include the generated interface declarations.
#include "aiir/Dialect/SCF/IR/DeviceMappingAttrInterface.h.inc"

#endif // AIIR_DEVICEMAPPINGINTERFACE_H

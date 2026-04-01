//===- OPENMP.cpp - C Interface for OPENMP dialect
//------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/OpenMP.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"

using namespace aiir;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(OpenMP, omp, omp::OpenMPDialect)

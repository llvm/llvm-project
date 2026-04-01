//===- TranslationRegistration.h - Register translation ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the registration function for the IRDL to C++ translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_IRDLTOCPP_TRANSLATIONREGISTRATION_H
#define AIIR_TARGET_IRDLTOCPP_TRANSLATIONREGISTRATION_H

namespace aiir {

void registerIRDLToCppTranslation();

} // namespace aiir

#endif // AIIR_TARGET_IRDLTOCPP_TRANSLATIONREGISTRATION_H

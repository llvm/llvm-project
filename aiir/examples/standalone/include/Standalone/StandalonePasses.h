//===- StandalonePasses.h - Standalone passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef STANDALONE_STANDALONEPASSES_H
#define STANDALONE_STANDALONEPASSES_H

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "aiir/Pass/Pass.h"
#include <memory>

namespace aiir {
namespace standalone {
#define GEN_PASS_DECL
#include "Standalone/StandalonePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Standalone/StandalonePasses.h.inc"
} // namespace standalone
} // namespace aiir

#endif

//===- TestInterfaces.h - AIIR interfaces for testing -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares interfaces for the 'test' dialect that can be used for
// testing the interface infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TEST_LIB_DIALECT_TEST_TESTINTERFACES_H
#define AIIR_TEST_LIB_DIALECT_TEST_TESTINTERFACES_H

#include "aiir/Interfaces/SideEffectInterfaces.h"

namespace aiir {
namespace TestEffects {
struct Effect : public SideEffects::Effect {
  using SideEffects::Effect::Effect;

  template <typename Derived>
  using Base = SideEffects::Effect::Base<Derived, Effect>;

  static bool classof(const SideEffects::Effect *effect);
};

using EffectInstance = SideEffects::EffectInstance<Effect>;

struct Concrete : public Effect::Base<Concrete> {};

} // namespace TestEffects
} // namespace aiir

#include "TestOpInterfaces.h.inc"

#endif // AIIR_TEST_LIB_DIALECT_TEST_TESTINTERFACES_H

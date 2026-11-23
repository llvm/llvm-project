//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBInitializationOptions.h"
#include "lldb/Initialization/SystemInitializer.h"
#include "lldb/Utility/Instrumentation.h"

#include "Utils.h"

using namespace lldb;
using namespace lldb_private;

SBInitializationOptions::SBInitializationOptions()
    : m_opaque_up(std::make_unique<InitializationOptions>()) {
  LLDB_INSTRUMENT_VA(this);
}

SBInitializationOptions::SBInitializationOptions(
    const SBInitializationOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBInitializationOptions::~SBInitializationOptions() = default;

const SBInitializationOptions &
SBInitializationOptions::operator=(const SBInitializationOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

const InitializationOptions &SBInitializationOptions::ref() const {
  return *m_opaque_up;
}

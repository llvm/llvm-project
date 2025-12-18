//===-- SBVariableAnnotator.cpp
//-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBVariableAnnotator.h"
#include "lldb/Utility/Instrumentation.h"

using namespace lldb;
using namespace lldb_private;

SBVariableAnnotator::SBVariableAnnotator() { LLDB_INSTRUMENT_VA(this); }

SBVariableAnnotator::SBVariableAnnotator(const SBVariableAnnotator &rhs) {
  LLDB_INSTRUMENT_VA(this);
}

const SBVariableAnnotator &
SBVariableAnnotator::operator=(const SBVariableAnnotator &rhs) {
  LLDB_INSTRUMENT_VA(this);

  //   if (this != &rhs)
  //     // TODO
  return *this;
}

SBVariableAnnotator::~SBVariableAnnotator() = default;

SBVariableAnnotator::operator bool() const {
  LLDB_INSTRUMENT_VA(this);

  return IsValid();
}

bool lldb::SBVariableAnnotator::IsValid() const { return false; }

//===- bolt/Profile/ProfileReaderBase.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Interface to be implemented by all profile readers.
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/ProfileReaderBase.h"

namespace llvm {
namespace bolt {

bool ProfileReaderBase::mayHaveProfileData(const BinaryFunction &BF) {
  return true;
}

} // namespace bolt
} // namespace llvm

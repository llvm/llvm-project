//===----------------------- InitMap.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "InitMap.h"

using namespace clang;
using namespace clang::interp;

bool InitMap::initializeElement(unsigned I) {
  unsigned Bucket = I / PER_FIELD;
  T Mask = T(1) << (I % PER_FIELD);
  if (!(data()[Bucket] & Mask)) {
    data()[Bucket] |= Mask;
    UninitFields -= 1;
  }
  return UninitFields == 0;
}

bool InitMap::isElementInitialized(unsigned I) const {
  if (UninitFields == 0)
    return true;
  unsigned Bucket = I / PER_FIELD;
  return data()[Bucket] & (T(1) << (I % PER_FIELD));
}

// Values in the second half of data() are inverted,
// i.e. 0 means "lifetime started".
void InitMap::startElementLifetime(unsigned I) {
  unsigned LifetimeIndex = NumElems + I;

  unsigned Bucket = numFields(NumElems) / 2 + (I / PER_FIELD);
  T Mask = T(1) << (LifetimeIndex % PER_FIELD);
  if ((data()[Bucket] & Mask)) {
    data()[Bucket] &= ~Mask;
    --DeadFields;
  }
}

// Values in the second half of data() are inverted,
// i.e. 0 means "lifetime started".
void InitMap::endElementLifetime(unsigned I) {
  unsigned LifetimeIndex = NumElems + I;

  unsigned Bucket = numFields(NumElems) / 2 + (I / PER_FIELD);
  T Mask = T(1) << (LifetimeIndex % PER_FIELD);
  if (!(data()[Bucket] & Mask)) {
    data()[Bucket] |= Mask;
    ++DeadFields;
  }
}

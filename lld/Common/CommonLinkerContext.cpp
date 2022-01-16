//===- CommonLinkerContext.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"

using namespace llvm;
using namespace lld;

// Reference to the current LLD instance.
static CommonLinkerContext *lctx;

CommonLinkerContext::CommonLinkerContext() { lctx = this; }

CommonLinkerContext::~CommonLinkerContext() {
  assert(lctx);
  // Explicitly call the destructors since we created the objects with placement
  // new in SpecificAlloc::create().
  for (auto &it : instances)
    it.second->~SpecificAllocBase();
  lctx = nullptr;
}

CommonLinkerContext &lld::commonContext() {
  assert(lctx);
  return *lctx;
}

bool lld::hasContext() { return lctx != nullptr; }

void CommonLinkerContext::destroy() {
  if (lctx == nullptr)
    return;
  delete lctx;
}

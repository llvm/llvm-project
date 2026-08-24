//===-- Unittests for the Emissary host handler registry ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/emissary_rpc_server.h"
#include "test/UnitTest/Test.h"

namespace {

// Distinct handler bodies so lookups can be told apart by return value. The
// bodies are never actually executed against a real RPC buffer here; the tests
// only compare function pointers and invoke them with a stub descriptor.
EmissaryReturn_t handlerA(char *, emisArgBuf_t *, emis_argptr_t *[]) {
  return 0xA;
}
EmissaryReturn_t handlerB(char *, emisArgBuf_t *, emis_argptr_t *[]) {
  return 0xB;
}

// Use ids near the top of the range to avoid colliding with real
// offload_emis_id_t values a co-linked client might register.
constexpr unsigned int kIdA = EMISSARY_MAX_REGISTERED_IDS - 2;
constexpr unsigned int kIdB = EMISSARY_MAX_REGISTERED_IDS - 3;

} // namespace

TEST(LlvmLibcEmissaryRegistryTest, LookupUnregisteredIsNull) {
  EXPECT_EQ(EmissaryLookup(kIdA), static_cast<EmissaryHandler_t>(nullptr));
}

TEST(LlvmLibcEmissaryRegistryTest, RegisterThenLookup) {
  ASSERT_TRUE(EmissaryRegister(kIdA, &handlerA));
  EmissaryHandler_t Got = EmissaryLookup(kIdA);
  ASSERT_TRUE(Got == &handlerA);

  // The looked-up handler is really callable and is the one we stored.
  emisArgBuf_t Ab = {};
  EXPECT_EQ(Got(nullptr, &Ab, nullptr), static_cast<EmissaryReturn_t>(0xA));
}

TEST(LlvmLibcEmissaryRegistryTest, IdempotentReregisterSucceeds) {
  ASSERT_TRUE(EmissaryRegister(kIdB, &handlerB));
  // Same id, same handler: allowed.
  EXPECT_TRUE(EmissaryRegister(kIdB, &handlerB));
  // Same id, different handler: rejected, original preserved.
  EXPECT_FALSE(EmissaryRegister(kIdB, &handlerA));
  EXPECT_TRUE(EmissaryLookup(kIdB) == &handlerB);
}

TEST(LlvmLibcEmissaryRegistryTest, RejectsNullHandler) {
  EXPECT_FALSE(EmissaryRegister(EMISSARY_MAX_REGISTERED_IDS - 4, nullptr));
}

TEST(LlvmLibcEmissaryRegistryTest, RejectsOutOfRangeId) {
  EXPECT_FALSE(EmissaryRegister(EMISSARY_MAX_REGISTERED_IDS, &handlerA));
  EXPECT_FALSE(EmissaryRegister(EMISSARY_MAX_REGISTERED_IDS + 100, &handlerA));
  EXPECT_EQ(EmissaryLookup(EMISSARY_MAX_REGISTERED_IDS),
            static_cast<EmissaryHandler_t>(nullptr));
}

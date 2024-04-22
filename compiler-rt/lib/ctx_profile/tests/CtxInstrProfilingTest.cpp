#include "../CtxInstrProfiling.h"
#include "gtest/gtest.h"

using namespace __ctx_profile;

TEST(ArenaTest, Basic) {
  Arena *A = Arena::allocateNewArena(1024);
  EXPECT_EQ(A->size(), 1024U);
  EXPECT_EQ(A->next(), nullptr);

  auto *M1 = A->tryBumpAllocate(1020);
  EXPECT_NE(M1, nullptr);
  auto *M2 = A->tryBumpAllocate(4);
  EXPECT_NE(M2, nullptr);
  EXPECT_EQ(M1 + 1020, M2);
  EXPECT_EQ(A->tryBumpAllocate(1), nullptr);
  Arena *A2 = Arena::allocateNewArena(2024, A);
  EXPECT_EQ(A->next(), A2);
  EXPECT_EQ(A2->next(), nullptr);
  Arena::freeArenaList(A);
  EXPECT_EQ(A, nullptr);
}

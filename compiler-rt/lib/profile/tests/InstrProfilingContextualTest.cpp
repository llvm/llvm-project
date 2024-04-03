#include "../InstrProfilingContextual.h"
#include "gtest/gtest.h"
#include <mutex>
#include <thread>

using namespace __profile;

TEST(ArenaTest, Basic) {
  Arena * A = Arena::allocate(1024);
  EXPECT_EQ(A->size(), 1024U);
  EXPECT_EQ(A->next(), nullptr);

  auto *M1 = A->tryAllocate(1020); 
  EXPECT_NE(M1, nullptr);
  auto *M2 = A->tryAllocate(4);
  EXPECT_NE(M2, nullptr);
  EXPECT_EQ(M1 + 1020, M2);
  EXPECT_EQ(A->tryAllocate(1), nullptr);
  Arena *A2 = Arena::allocate(2024, A);
  EXPECT_EQ(A->next(), A2);
  EXPECT_EQ(A2->next(), nullptr);
}

TEST(ContextTest, Basic) {
  ContextRoot Root;
  memset(&Root, 0, sizeof(ContextRoot));
  auto *Ctx = __llvm_instrprof_start_context(&Root, 1, 10, 4);
  EXPECT_NE(Ctx, nullptr);
  EXPECT_NE(Root.CurrentMem, nullptr);
  EXPECT_EQ(Root.FirstMemBlock, Root.CurrentMem);
  EXPECT_EQ(Ctx->size(), sizeof(ContextNode) + 10 * sizeof(uint64_t) +
                             4 * sizeof(ContextNode *));
  EXPECT_EQ(Ctx->counters_size(), 10U);
  EXPECT_EQ(Ctx->callsites_size(), 4U);
  EXPECT_EQ(__llvm_instrprof_current_context_root, &Root);
  Root.Taken.CheckLocked();
  EXPECT_FALSE(Root.Taken.TryLock());
  __llvm_instrprof_release_context(&Root);
  EXPECT_EQ(__llvm_instrprof_current_context_root, nullptr);
  EXPECT_TRUE(Root.Taken.TryLock());
  Root.Taken.Unlock();
}

TEST(ContextTest, Callsite) {
  ContextRoot Root;
  memset(&Root, 0, sizeof(ContextRoot));
  auto *Ctx = __llvm_instrprof_start_context(&Root, 1, 10, 4);
  int OpaqueValue = 0;
  const bool IsScratch = isScratch(Ctx);
  EXPECT_FALSE(IsScratch);
  __llvm_instrprof_expected_callee[0] = &OpaqueValue;
  __llvm_instrprof_callsite[0] = &Ctx->subContexts()[2];
  auto *Subctx = __llvm_instrprof_get_context(&OpaqueValue, 2, 3, 1);
  EXPECT_EQ(Ctx->subContexts()[2], Subctx);
  EXPECT_EQ(Subctx->counters_size(), 3U);
  EXPECT_EQ(Subctx->callsites_size(), 1U);
  EXPECT_EQ(__llvm_instrprof_expected_callee[0], nullptr);
  EXPECT_EQ(__llvm_instrprof_callsite[0], nullptr);
  
  EXPECT_EQ(Subctx->size(), sizeof(ContextNode) + 3 * sizeof(uint64_t) +
                                1 * sizeof(ContextNode *));
  __llvm_instrprof_release_context(&Root);
}

TEST(ContextTest, ScratchNoCollection) {
  EXPECT_EQ(__llvm_instrprof_current_context_root, nullptr);
  int OpaqueValue = 0;
  // this would be the very first function executing this. the TLS is empty,
  // too.
  auto *Ctx = __llvm_instrprof_get_context(&OpaqueValue, 2, 3, 1);
  EXPECT_TRUE(isScratch(Ctx));
}

TEST(ContextTest, ScratchDuringCollection) {
  ContextRoot Root;
  memset(&Root, 0, sizeof(ContextRoot));
  auto *Ctx = __llvm_instrprof_start_context(&Root, 1, 10, 4);
  int OpaqueValue = 0;
  int OtherOpaqueValue = 0;
  __llvm_instrprof_expected_callee[0] = &OpaqueValue;
  __llvm_instrprof_callsite[0] = &Ctx->subContexts()[2];
  auto *Subctx = __llvm_instrprof_get_context(&OtherOpaqueValue, 2, 3, 1);
  EXPECT_TRUE(isScratch(Subctx));
  EXPECT_EQ(__llvm_instrprof_expected_callee[0], nullptr);
  EXPECT_EQ(__llvm_instrprof_callsite[0], nullptr);
  
  int ThirdOpaqueValue = 0;
  __llvm_instrprof_expected_callee[1] = &ThirdOpaqueValue;
  __llvm_instrprof_callsite[1] = &Subctx->subContexts()[0];

  auto *Subctx2 = __llvm_instrprof_get_context(&ThirdOpaqueValue, 3, 0, 0);
  EXPECT_TRUE(isScratch(Subctx2));
  
  __llvm_instrprof_release_context(&Root);
}

TEST(ContextTest, ConcurrentRootCollection) {
  ContextRoot Root;
  memset(&Root, 0, sizeof(ContextRoot));
  std::atomic<int> NonScratch = 0;
  std::atomic<int> Executions = 0;

  __sanitizer::Semaphore GotCtx;

  auto Entrypoint = [&]() {
    ++Executions;
    auto *Ctx = __llvm_instrprof_start_context(&Root, 1, 10, 4);
    GotCtx.Post();
    const bool IS = isScratch(Ctx);
    NonScratch += (!IS);
    if (!IS) {
      GotCtx.Wait();
      GotCtx.Wait();
    }
    __llvm_instrprof_release_context(&Root);
  };
  std::thread T1(Entrypoint);
  std::thread T2(Entrypoint);
  T1.join();
  T2.join();
  EXPECT_EQ(NonScratch, 1);
  EXPECT_EQ(Executions, 2);
}
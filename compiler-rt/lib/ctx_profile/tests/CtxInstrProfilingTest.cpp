#include "../CtxInstrProfiling.h"
#include "gtest/gtest.h"
#include <thread>

using namespace __ctx_profile;

class ContextTest : public ::testing::Test {
  void SetUp() override { memset(&Root, 0, sizeof(ContextRoot)); }
  void TearDown() override { __llvm_ctx_profile_free(); }

public:
  ContextRoot Root;
};

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

TEST_F(ContextTest, Basic) {
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  EXPECT_NE(Ctx, nullptr);
  EXPECT_NE(Root.CurrentMem, nullptr);
  EXPECT_EQ(Root.FirstMemBlock, Root.CurrentMem);
  EXPECT_EQ(Ctx->size(), sizeof(ContextNode) + 10 * sizeof(uint64_t) +
                             4 * sizeof(ContextNode *));
  EXPECT_EQ(Ctx->counters_size(), 10U);
  EXPECT_EQ(Ctx->callsites_size(), 4U);
  EXPECT_EQ(__llvm_ctx_profile_current_context_root, &Root);
  Root.Taken.CheckLocked();
  EXPECT_FALSE(Root.Taken.TryLock());
  __llvm_ctx_profile_release_context(&Root);
  EXPECT_EQ(__llvm_ctx_profile_current_context_root, nullptr);
  EXPECT_TRUE(Root.Taken.TryLock());
  Root.Taken.Unlock();
}

TEST_F(ContextTest, Callsite) {
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  int OpaqueValue = 0;
  const bool IsScratch = isScratch(Ctx);
  EXPECT_FALSE(IsScratch);
  // This is the sequence the caller performs - it's the lowering of the
  // instrumentation of the callsite "2". "2" is arbitrary here.
  __llvm_ctx_profile_expected_callee[0] = &OpaqueValue;
  __llvm_ctx_profile_callsite[0] = &Ctx->subContexts()[2];
  // This is what the callee does
  auto *Subctx = __llvm_ctx_profile_get_context(&OpaqueValue, 2, 3, 1);
  // We expect the subcontext to be appropriately placed and dimensioned
  EXPECT_EQ(Ctx->subContexts()[2], Subctx);
  EXPECT_EQ(Subctx->counters_size(), 3U);
  EXPECT_EQ(Subctx->callsites_size(), 1U);
  // We reset these in _get_context.
  EXPECT_EQ(__llvm_ctx_profile_expected_callee[0], nullptr);
  EXPECT_EQ(__llvm_ctx_profile_callsite[0], nullptr);

  EXPECT_EQ(Subctx->size(), sizeof(ContextNode) + 3 * sizeof(uint64_t) +
                                1 * sizeof(ContextNode *));
  __llvm_ctx_profile_release_context(&Root);
}

TEST_F(ContextTest, ScratchNoCollection) {
  EXPECT_EQ(__llvm_ctx_profile_current_context_root, nullptr);
  int OpaqueValue = 0;
  // this would be the very first function executing this. the TLS is empty,
  // too.
  auto *Ctx = __llvm_ctx_profile_get_context(&OpaqueValue, 2, 3, 1);
  // We never entered a context (_start_context was never called) - so the
  // returned context must be scratch.
  EXPECT_TRUE(isScratch(Ctx));
}

TEST_F(ContextTest, ScratchDuringCollection) {
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  int OpaqueValue = 0;
  int OtherOpaqueValue = 0;
  __llvm_ctx_profile_expected_callee[0] = &OpaqueValue;
  __llvm_ctx_profile_callsite[0] = &Ctx->subContexts()[2];
  auto *Subctx = __llvm_ctx_profile_get_context(&OtherOpaqueValue, 2, 3, 1);
  // We expected a different callee - so return scratch. It mimics what happens
  // in the case of a signal handler - in this case, OtherOpaqueValue is the
  // signal handler.
  EXPECT_TRUE(isScratch(Subctx));
  EXPECT_EQ(__llvm_ctx_profile_expected_callee[0], nullptr);
  EXPECT_EQ(__llvm_ctx_profile_callsite[0], nullptr);

  int ThirdOpaqueValue = 0;
  __llvm_ctx_profile_expected_callee[1] = &ThirdOpaqueValue;
  __llvm_ctx_profile_callsite[1] = &Subctx->subContexts()[0];

  auto *Subctx2 = __llvm_ctx_profile_get_context(&ThirdOpaqueValue, 3, 0, 0);
  EXPECT_TRUE(isScratch(Subctx2));

  __llvm_ctx_profile_release_context(&Root);
}

TEST_F(ContextTest, NeedMoreMemory) {
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  int OpaqueValue = 0;
  const bool IsScratch = isScratch(Ctx);
  EXPECT_FALSE(IsScratch);
  const auto *CurrentMem = Root.CurrentMem;
  __llvm_ctx_profile_expected_callee[0] = &OpaqueValue;
  __llvm_ctx_profile_callsite[0] = &Ctx->subContexts()[2];
  // Allocate a massive subcontext to force new arena allocation
  auto *Subctx = __llvm_ctx_profile_get_context(&OpaqueValue, 3, 1 << 20, 1);
  EXPECT_EQ(Ctx->subContexts()[2], Subctx);
  EXPECT_NE(CurrentMem, Root.CurrentMem);
  EXPECT_NE(Root.CurrentMem, nullptr);
}

TEST_F(ContextTest, ConcurrentRootCollection) {
  std::atomic<int> NonScratch = 0;
  std::atomic<int> Executions = 0;

  __sanitizer::Semaphore GotCtx;

  auto Entrypoint = [&]() {
    ++Executions;
    auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
    GotCtx.Post();
    const bool IS = isScratch(Ctx);
    NonScratch += (!IS);
    if (!IS) {
      GotCtx.Wait();
      GotCtx.Wait();
    }
    __llvm_ctx_profile_release_context(&Root);
  };
  std::thread T1(Entrypoint);
  std::thread T2(Entrypoint);
  T1.join();
  T2.join();
  EXPECT_EQ(NonScratch, 1);
  EXPECT_EQ(Executions, 2);
}

TEST_F(ContextTest, Dump) {
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  int OpaqueValue = 0;
  __llvm_ctx_profile_expected_callee[0] = &OpaqueValue;
  __llvm_ctx_profile_callsite[0] = &Ctx->subContexts()[2];
  auto *Subctx = __llvm_ctx_profile_get_context(&OpaqueValue, 2, 3, 1);
  (void)Subctx;
  __llvm_ctx_profile_release_context(&Root);

  struct Writer {
    ContextRoot *const Root;
    const size_t Entries;
    bool State = false;
    Writer(ContextRoot *Root, size_t Entries) : Root(Root), Entries(Entries) {}

    bool write(const ContextNode &Node) {
      EXPECT_FALSE(Root->Taken.TryLock());
      EXPECT_EQ(Node.guid(), 1);
      EXPECT_EQ(Node.counters()[0], Entries);
      EXPECT_EQ(Node.counters_size(), 10);
      EXPECT_EQ(Node.callsites_size(), 4);
      EXPECT_EQ(Node.subContexts()[0], nullptr);
      EXPECT_EQ(Node.subContexts()[1], nullptr);
      EXPECT_NE(Node.subContexts()[2], nullptr);
      EXPECT_EQ(Node.subContexts()[3], nullptr);
      const auto &SN = *Node.subContexts()[2];
      EXPECT_EQ(SN.guid(), 2);
      EXPECT_EQ(SN.counters()[0], Entries);
      EXPECT_EQ(SN.counters_size(), 3);
      EXPECT_EQ(SN.callsites_size(), 1);
      EXPECT_EQ(SN.subContexts()[0], nullptr);
      State = true;
      return true;
    }
  };
  Writer W(&Root, 1);
  EXPECT_FALSE(W.State);
  __llvm_ctx_profile_fetch(&W, [](void *W, const ContextNode &Node) -> bool {
    return reinterpret_cast<Writer *>(W)->write(Node);
  });
  EXPECT_TRUE(W.State);

  // this resets all counters but not the internal structure.
  __llvm_ctx_profile_start_collection();
  Writer W2(&Root, 0);
  EXPECT_FALSE(W2.State);
  __llvm_ctx_profile_fetch(&W2, [](void *W, const ContextNode &Node) -> bool {
    return reinterpret_cast<Writer *>(W)->write(Node);
  });
  EXPECT_TRUE(W2.State);
}

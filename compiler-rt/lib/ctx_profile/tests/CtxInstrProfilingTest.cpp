#include "../CtxInstrProfiling.h"
#include "gtest/gtest.h"
#include <thread>

using namespace __ctx_profile;

class ContextTest : public ::testing::Test {
  void SetUp() override { Root.getOrAllocateContextRoot(); }
  void TearDown() override { __llvm_ctx_profile_free(); }

public:
  FunctionData Root;
};

TEST(ArenaTest, ZeroInit) {
  char Buffer[1024];
  memset(Buffer, 1, 1024);
  Arena *A = new (Buffer) Arena(10);
  for (auto I = 0U; I < A->size(); ++I)
    EXPECT_EQ(A->pos()[I], static_cast<char>(0));
  EXPECT_EQ(A->size(), 10U);
}

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
  __llvm_ctx_profile_start_collection();
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  ASSERT_NE(Ctx, nullptr);
  auto &CtxRoot = *Root.CtxRoot;
  EXPECT_NE(CtxRoot.CurrentMem, nullptr);
  EXPECT_EQ(CtxRoot.FirstMemBlock, CtxRoot.CurrentMem);
  EXPECT_EQ(Ctx->size(), sizeof(ContextNode) + 10 * sizeof(uint64_t) +
                             4 * sizeof(ContextNode *));
  EXPECT_EQ(Ctx->counters_size(), 10U);
  EXPECT_EQ(Ctx->callsites_size(), 4U);
  EXPECT_EQ(__llvm_ctx_profile_current_context_root, &CtxRoot);
  CtxRoot.Taken.CheckLocked();
  EXPECT_FALSE(CtxRoot.Taken.TryLock());
  __llvm_ctx_profile_release_context(&Root);
  EXPECT_EQ(__llvm_ctx_profile_current_context_root, nullptr);
  EXPECT_TRUE(CtxRoot.Taken.TryLock());
  CtxRoot.Taken.Unlock();
}

TEST_F(ContextTest, Callsite) {
  __llvm_ctx_profile_start_collection();
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  int FakeCalleeAddress = 0;
  const bool IsScratch = isScratch(Ctx);
  EXPECT_FALSE(IsScratch);
  // This is the sequence the caller performs - it's the lowering of the
  // instrumentation of the callsite "2". "2" is arbitrary here.
  __llvm_ctx_profile_expected_callee[0] = &FakeCalleeAddress;
  __llvm_ctx_profile_callsite[0] = &Ctx->subContexts()[2];
  // This is what the callee does
  FunctionData FData;
  auto *Subctx =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
  // This should not have required creating a flat context.
  EXPECT_EQ(FData.FlatCtx, nullptr);
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

TEST_F(ContextTest, ScratchNoCollectionProfilingNotStarted) {
  // This test intentionally does not call __llvm_ctx_profile_start_collection.
  EXPECT_EQ(__llvm_ctx_profile_current_context_root, nullptr);
  int FakeCalleeAddress = 0;
  // this would be the very first function executing this. the TLS is empty,
  // too.
  FunctionData FData;
  auto *Ctx =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
  // We never entered a context (_start_context was never called) - so the
  // returned context must be a tagged pointer.
  EXPECT_TRUE(isScratch(Ctx));
  // Because we didn't start collection, no flat profile should have been
  // allocated.
  EXPECT_EQ(FData.FlatCtx, nullptr);
}

TEST_F(ContextTest, ScratchNoCollectionProfilingStarted) {
  ASSERT_EQ(__llvm_ctx_profile_current_context_root, nullptr);
  int FakeCalleeAddress = 0;
  // Start collection, so the function gets a flat profile instead of scratch.
  __llvm_ctx_profile_start_collection();
  // this would be the very first function executing this. the TLS is empty,
  // too.
  FunctionData FData;
  auto *Ctx =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
  // We never entered a context (_start_context was never called) - so the
  // returned context must be a tagged pointer.
  EXPECT_TRUE(isScratch(Ctx));
  // Because we never entered a context, we should have allocated a flat context
  EXPECT_NE(FData.FlatCtx, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(FData.FlatCtx) + 1,
            reinterpret_cast<uintptr_t>(Ctx));
}

TEST_F(ContextTest, ScratchDuringCollection) {
  __llvm_ctx_profile_start_collection();
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  int FakeCalleeAddress = 0;
  int OtherFakeCalleeAddress = 0;
  __llvm_ctx_profile_expected_callee[0] = &FakeCalleeAddress;
  __llvm_ctx_profile_callsite[0] = &Ctx->subContexts()[2];
  FunctionData FData[3];
  auto *Subctx = __llvm_ctx_profile_get_context(
      &FData[0], &OtherFakeCalleeAddress, 2, 3, 1);
  // We expected a different callee - so return scratch. It mimics what happens
  // in the case of a signal handler - in this case, OtherFakeCalleeAddress is
  // the signal handler.
  EXPECT_TRUE(isScratch(Subctx));
  // We shouldn't have tried to return a flat context because we're under a
  // root.
  EXPECT_EQ(FData[0].FlatCtx, nullptr);
  EXPECT_EQ(__llvm_ctx_profile_expected_callee[0], nullptr);
  EXPECT_EQ(__llvm_ctx_profile_callsite[0], nullptr);

  int ThirdFakeCalleeAddress = 0;
  __llvm_ctx_profile_expected_callee[1] = &ThirdFakeCalleeAddress;
  __llvm_ctx_profile_callsite[1] = &Subctx->subContexts()[0];

  auto *Subctx2 = __llvm_ctx_profile_get_context(
      &FData[1], &ThirdFakeCalleeAddress, 3, 0, 0);
  // We again expect scratch because the '0' position is where the runtime
  // looks, so it doesn't matter the '1' position is populated correctly.
  EXPECT_TRUE(isScratch(Subctx2));
  EXPECT_EQ(FData[1].FlatCtx, nullptr);

  __llvm_ctx_profile_expected_callee[0] = &ThirdFakeCalleeAddress;
  __llvm_ctx_profile_callsite[0] = &Subctx->subContexts()[0];
  auto *Subctx3 = __llvm_ctx_profile_get_context(
      &FData[2], &ThirdFakeCalleeAddress, 3, 0, 0);
  // We expect scratch here, too, because the value placed in
  // __llvm_ctx_profile_callsite is scratch
  EXPECT_TRUE(isScratch(Subctx3));
  EXPECT_EQ(FData[2].FlatCtx, nullptr);

  __llvm_ctx_profile_release_context(&Root);
}

TEST_F(ContextTest, NeedMoreMemory) {
  __llvm_ctx_profile_start_collection();
  auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  int FakeCalleeAddress = 0;
  const bool IsScratch = isScratch(Ctx);
  EXPECT_FALSE(IsScratch);
  auto &CtxRoot = *Root.CtxRoot;
  const auto *CurrentMem = CtxRoot.CurrentMem;
  __llvm_ctx_profile_expected_callee[0] = &FakeCalleeAddress;
  __llvm_ctx_profile_callsite[0] = &Ctx->subContexts()[2];
  FunctionData FData;
  // Allocate a massive subcontext to force new arena allocation
  auto *Subctx =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 3, 1 << 20, 1);
  EXPECT_EQ(FData.FlatCtx, nullptr);
  EXPECT_EQ(Ctx->subContexts()[2], Subctx);
  EXPECT_NE(CurrentMem, CtxRoot.CurrentMem);
  EXPECT_NE(CtxRoot.CurrentMem, nullptr);
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
  int FakeCalleeAddress = 0;
  __llvm_ctx_profile_expected_callee[0] = &FakeCalleeAddress;
  __llvm_ctx_profile_callsite[0] = &Ctx->subContexts()[2];
  FunctionData FData;
  auto *Subctx =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
  (void)Subctx;
  __llvm_ctx_profile_release_context(&Root);

  class TestProfileWriter : public ProfileWriter {
  public:
    ContextRoot *const Root;
    const size_t Entries;

    int EnteredSectionCount = 0;
    int ExitedSectionCount = 0;
    int EnteredFlatCount = 0;
    int ExitedFlatCount = 0;
    int FlatsWritten = 0;

    bool State = false;

    TestProfileWriter(ContextRoot *Root, size_t Entries)
        : Root(Root), Entries(Entries) {}

    void writeContextual(const ContextNode &Node, const ContextNode *Unhandled,
                         uint64_t TotalRootEntryCount) override {
      EXPECT_EQ(TotalRootEntryCount, Entries);
      EXPECT_EQ(EnteredSectionCount, 1);
      EXPECT_EQ(ExitedSectionCount, 0);
      EXPECT_FALSE(Root->Taken.TryLock());
      EXPECT_EQ(Node.guid(), 1U);
      EXPECT_EQ(Node.counters()[0], Entries);
      EXPECT_EQ(Node.counters_size(), 10U);
      EXPECT_EQ(Node.callsites_size(), 4U);
      EXPECT_EQ(Node.subContexts()[0], nullptr);
      EXPECT_EQ(Node.subContexts()[1], nullptr);
      EXPECT_NE(Node.subContexts()[2], nullptr);
      EXPECT_EQ(Node.subContexts()[3], nullptr);
      const auto &SN = *Node.subContexts()[2];
      EXPECT_EQ(SN.guid(), 2U);
      EXPECT_EQ(SN.counters()[0], Entries);
      EXPECT_EQ(SN.counters_size(), 3U);
      EXPECT_EQ(SN.callsites_size(), 1U);
      EXPECT_EQ(SN.subContexts()[0], nullptr);
      State = true;
    }
    void startContextSection() override { ++EnteredSectionCount; }
    void endContextSection() override {
      EXPECT_EQ(EnteredSectionCount, 1);
      ++ExitedSectionCount;
    }
    void startFlatSection() override { ++EnteredFlatCount; }
    void writeFlat(GUID Guid, const uint64_t *Buffer,
                   size_t BufferSize) override {
      ++FlatsWritten;
      EXPECT_EQ(BufferSize, 3U);
      EXPECT_EQ(Buffer[0], 15U);
      EXPECT_EQ(Buffer[1], 0U);
      EXPECT_EQ(Buffer[2], 0U);
    }
    void endFlatSection() override { ++ExitedFlatCount; }
  };

  TestProfileWriter W(Root.CtxRoot, 1);
  EXPECT_FALSE(W.State);
  __llvm_ctx_profile_fetch(W);
  EXPECT_TRUE(W.State);

  // this resets all counters but not the internal structure.
  __llvm_ctx_profile_start_collection();
  auto *Flat =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
  (void)Flat;
  EXPECT_NE(FData.FlatCtx, nullptr);
  FData.FlatCtx->counters()[0] = 15U;
  TestProfileWriter W2(Root.CtxRoot, 0);
  EXPECT_FALSE(W2.State);
  __llvm_ctx_profile_fetch(W2);
  EXPECT_TRUE(W2.State);
  EXPECT_EQ(W2.EnteredSectionCount, 1);
  EXPECT_EQ(W2.ExitedSectionCount, 1);
  EXPECT_EQ(W2.EnteredFlatCount, 1);
  EXPECT_EQ(W2.FlatsWritten, 1);
  EXPECT_EQ(W2.ExitedFlatCount, 1);
}

TEST_F(ContextTest, MustNotBeRoot) {
  FunctionData FData;
  FData.CtxRoot = reinterpret_cast<ContextRoot *>(1U);
  int FakeCalleeAddress = 0;
  __llvm_ctx_profile_start_collection();
  auto *Subctx =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
  EXPECT_TRUE(isScratch(Subctx));
}

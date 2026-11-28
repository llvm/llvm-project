#include "../CtxInstrProfiling.h"
#include "gtest/gtest.h"
#include <thread>
#include <utility>
#include <vector>

using namespace __ctx_profile;

namespace {
class FlatCounterWriter : public ProfileWriter {
public:
  int FlatsWritten = 0;
  std::vector<std::pair<GUID, std::vector<uint64_t>>> FlatProfiles;

  void startContextSection() override {}
  void writeContextual(const ContextNode &, const ContextNode *,
                       uint64_t) override {}
  void endContextSection() override {}
  void startFlatSection() override {}
  void writeFlat(GUID Guid, const uint64_t *Buffer,
                 size_t BufferSize) override {
    ++FlatsWritten;
    FlatProfiles.emplace_back(
        Guid, std::vector<uint64_t>(Buffer, Buffer + BufferSize));
  }
  void endFlatSection() override {}
};
} // namespace

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

TEST_F(ContextTest, FlatProfileAfterContendedRootExit) {
  __llvm_ctx_profile_start_collection();
  __sanitizer::Semaphore RootLocked;
  __sanitizer::Semaphore ReleaseHolder;

  std::thread Holder([&]() {
    auto *Ctx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
    EXPECT_FALSE(isScratch(Ctx));
    RootLocked.Post();
    ReleaseHolder.Wait();
    __llvm_ctx_profile_release_context(&Root);
  });

  RootLocked.Wait();
  auto *ContendedCtx = __llvm_ctx_profile_start_context(&Root, 1, 10, 4);
  EXPECT_TRUE(isScratch(ContendedCtx));
  EXPECT_EQ(__llvm_ctx_profile_current_context_root, nullptr);

  int CalleeAddr1 = 0;
  FunctionData UnderRootCallee;
  auto *CalleeCtx =
      __llvm_ctx_profile_get_context(&UnderRootCallee, &CalleeAddr1, 2, 3, 0);
  EXPECT_TRUE(isScratch(CalleeCtx));
  EXPECT_EQ(UnderRootCallee.FlatCtx, nullptr);

  // Releasing a non-root or musttail function must not clear under-context
  // state.
  __llvm_ctx_profile_release_context(&UnderRootCallee);
  FunctionData MusttailFunc;
  MusttailFunc.CtxRoot = reinterpret_cast<ContextRoot *>(1U);
  __llvm_ctx_profile_release_context(&MusttailFunc);

  int CalleeAddr2 = 0;
  FunctionData StillUnderRootCallee;
  auto *StillUnderCtx = __llvm_ctx_profile_get_context(&StillUnderRootCallee,
                                                       &CalleeAddr2, 3, 3, 0);
  EXPECT_TRUE(isScratch(StillUnderCtx));
  EXPECT_EQ(StillUnderRootCallee.FlatCtx, nullptr);

  __llvm_ctx_profile_release_context(&Root);

  int CalleeAddr3 = 0;
  FunctionData PostRootFunc;
  auto *PostRootCtx =
      __llvm_ctx_profile_get_context(&PostRootFunc, &CalleeAddr3, 4, 3, 0);
  EXPECT_TRUE(isScratch(PostRootCtx));
  ASSERT_NE(PostRootFunc.FlatCtx, nullptr);
  EXPECT_EQ(PostRootFunc.FlatCtx->entrycount(), 1U);

  // Also verify contended autodetected root entry via get_context.
  int RootAddr = 0;
  auto *AutoContendedCtx =
      __llvm_ctx_profile_get_context(&Root, &RootAddr, 1, 10, 4);
  EXPECT_TRUE(isScratch(AutoContendedCtx));
  EXPECT_EQ(__llvm_ctx_profile_current_context_root, nullptr);

  int CalleeAddr4 = 0;
  FunctionData UnderAutoRootCallee;
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_get_context(&UnderAutoRootCallee,
                                                       &CalleeAddr4, 5, 3, 0)));
  EXPECT_EQ(UnderAutoRootCallee.FlatCtx, nullptr);

  __llvm_ctx_profile_release_context(&Root);

  int CalleeAddr5 = 0;
  FunctionData PostAutoRootFunc;
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_get_context(&PostAutoRootFunc,
                                                       &CalleeAddr5, 6, 3, 0)));
  ASSERT_NE(PostAutoRootFunc.FlatCtx, nullptr);
  EXPECT_EQ(PostAutoRootFunc.FlatCtx->entrycount(), 1U);

  ReleaseHolder.Post();
  Holder.join();
}

TEST_F(ContextTest, FlatProfileUnderRecursiveAndNestedContendedRoots) {
  __llvm_ctx_profile_start_collection();

  // An unmatched root release (e.g. a function promoted while already on the
  // stack) must not underflow the depth counter.
  __llvm_ctx_profile_release_context(&Root);
  FunctionData DepthZeroFunc;
  int DepthZeroAddr = 0;
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_get_context(
      &DepthZeroFunc, &DepthZeroAddr, 10, 2, 0)));
  ASSERT_NE(DepthZeroFunc.FlatCtx, nullptr);
  EXPECT_EQ(DepthZeroFunc.FlatCtx->entrycount(), 1U);

  FunctionData Root2;
  Root2.getOrAllocateContextRoot();
  __sanitizer::Semaphore RootsLocked;
  __sanitizer::Semaphore ReleaseHolders;

  std::thread Holder1([&]() {
    EXPECT_FALSE(isScratch(__llvm_ctx_profile_start_context(&Root, 1, 10, 4)));
    RootsLocked.Post();
    ReleaseHolders.Wait();
    __llvm_ctx_profile_release_context(&Root);
  });
  std::thread Holder2([&]() {
    EXPECT_FALSE(isScratch(__llvm_ctx_profile_start_context(&Root2, 2, 10, 4)));
    RootsLocked.Post();
    ReleaseHolders.Wait();
    __llvm_ctx_profile_release_context(&Root2);
  });
  RootsLocked.Wait();
  RootsLocked.Wait();

  // Recursive contended root: flat profiling stays suppressed until the
  // outermost exit.
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_start_context(&Root, 1, 10, 4)));
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_start_context(&Root, 1, 10, 4)));
  __llvm_ctx_profile_release_context(&Root);

  FunctionData BetweenRecursiveExits;
  int Addr1 = 0;
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_get_context(&BetweenRecursiveExits,
                                                       &Addr1, 11, 2, 0)));
  EXPECT_EQ(BetweenRecursiveExits.FlatCtx, nullptr);

  __llvm_ctx_profile_release_context(&Root);
  FunctionData AfterRecursiveExit;
  int Addr2 = 0;
  EXPECT_TRUE(isScratch(
      __llvm_ctx_profile_get_context(&AfterRecursiveExit, &Addr2, 12, 2, 0)));
  ASSERT_NE(AfterRecursiveExit.FlatCtx, nullptr);
  EXPECT_EQ(AfterRecursiveExit.FlatCtx->entrycount(), 1U);

  // Nested contended roots: exiting the inner root keeps flat profiling
  // suppressed under the outer contended root.
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_start_context(&Root, 1, 10, 4)));
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_start_context(&Root2, 2, 10, 4)));
  __llvm_ctx_profile_release_context(&Root2);

  FunctionData BetweenNestedExits;
  int Addr3 = 0;
  EXPECT_TRUE(isScratch(
      __llvm_ctx_profile_get_context(&BetweenNestedExits, &Addr3, 13, 2, 0)));
  EXPECT_EQ(BetweenNestedExits.FlatCtx, nullptr);

  __llvm_ctx_profile_release_context(&Root);
  FunctionData AfterNestedExit;
  int Addr4 = 0;
  EXPECT_TRUE(isScratch(
      __llvm_ctx_profile_get_context(&AfterNestedExit, &Addr4, 14, 2, 0)));
  ASSERT_NE(AfterNestedExit.FlatCtx, nullptr);
  EXPECT_EQ(AfterNestedExit.FlatCtx->entrycount(), 1U);

  // Nested uncontended inner root under a contended outer root: exiting the
  // inner root still keeps flat profiling suppressed until the outer root
  // exits.
  FunctionData Root3;
  Root3.getOrAllocateContextRoot();
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_start_context(&Root, 1, 10, 4)));
  EXPECT_FALSE(isScratch(__llvm_ctx_profile_start_context(&Root3, 3, 10, 4)));
  __llvm_ctx_profile_release_context(&Root3);

  FunctionData UnderOuterAfterInnerUnlock;
  int Addr5 = 0;
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_get_context(
      &UnderOuterAfterInnerUnlock, &Addr5, 15, 2, 0)));
  EXPECT_EQ(UnderOuterAfterInnerUnlock.FlatCtx, nullptr);

  __llvm_ctx_profile_release_context(&Root);
  FunctionData AfterOuterExit;
  int Addr6 = 0;
  EXPECT_TRUE(isScratch(
      __llvm_ctx_profile_get_context(&AfterOuterExit, &Addr6, 16, 2, 0)));
  ASSERT_NE(AfterOuterExit.FlatCtx, nullptr);
  EXPECT_EQ(AfterOuterExit.FlatCtx->entrycount(), 1U);

  // Nested autodetected inner root entered via get_context under a held outer
  // root must balance UnderContextDepth so flat profiling stays suppressed even
  // if a subsequent contended root clears CurrentRoot.
  FunctionData HeldOuter;
  HeldOuter.getOrAllocateContextRoot();
  FunctionData AutoInner;
  AutoInner.getOrAllocateContextRoot();
  int AutoInnerAddr = 0;
  EXPECT_FALSE(
      isScratch(__llvm_ctx_profile_start_context(&HeldOuter, 4, 10, 4)));
  EXPECT_TRUE(isScratch(
      __llvm_ctx_profile_get_context(&AutoInner, &AutoInnerAddr, 5, 4, 0)));
  __llvm_ctx_profile_release_context(&AutoInner);
  EXPECT_TRUE(isScratch(__llvm_ctx_profile_start_context(&Root, 1, 10, 4)));
  __llvm_ctx_profile_release_context(&Root);

  FunctionData UnderHeldOuter;
  int Addr7 = 0;
  EXPECT_TRUE(isScratch(
      __llvm_ctx_profile_get_context(&UnderHeldOuter, &Addr7, 17, 2, 0)));
  EXPECT_EQ(UnderHeldOuter.FlatCtx, nullptr);

  __llvm_ctx_profile_release_context(&HeldOuter);
  FunctionData AfterHeldOuterExit;
  int Addr8 = 0;
  EXPECT_TRUE(isScratch(
      __llvm_ctx_profile_get_context(&AfterHeldOuterExit, &Addr8, 18, 2, 0)));
  ASSERT_NE(AfterHeldOuterExit.FlatCtx, nullptr);
  EXPECT_EQ(AfterHeldOuterExit.FlatCtx->entrycount(), 1U);

  ReleaseHolders.Post(2);
  Holder1.join();
  Holder2.join();
}

TEST_F(ContextTest, FlatProfileSkipsWhenMutexLocked) {
  __llvm_ctx_profile_start_collection();
  FunctionData FData;
  int FakeCalleeAddress = 0;
  {
    __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> Lock(
        &FData.Mutex);
    auto *Ctx =
        __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
    EXPECT_TRUE(isScratch(Ctx));
    EXPECT_EQ(FData.FlatCtx, nullptr);
  }

  FlatCounterWriter W1;
  testing::internal::CaptureStderr();
  EXPECT_TRUE(__llvm_ctx_profile_fetch(W1));
  std::string FetchErr1 = testing::internal::GetCapturedStderr();
  EXPECT_EQ(W1.FlatsWritten, 0);
  EXPECT_NE(FetchErr1.find("would have waited for a lock: 1\n"),
            std::string::npos);

  __llvm_ctx_profile_start_collection();
  auto *Ctx =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
  EXPECT_TRUE(isScratch(Ctx));
  ASSERT_NE(FData.FlatCtx, nullptr);
  EXPECT_EQ(FData.FlatCtx->entrycount(), 1U);

  // Once FlatCtx is allocated, callers read it without locking FData.Mutex.
  {
    __sanitizer::GenericScopedLock<__sanitizer::StaticSpinMutex> Lock(
        &FData.Mutex);
    Ctx = __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
    EXPECT_TRUE(isScratch(Ctx));
    EXPECT_EQ(FData.FlatCtx->entrycount(), 2U);
  }

  FlatCounterWriter W2;
  testing::internal::CaptureStderr();
  EXPECT_TRUE(__llvm_ctx_profile_fetch(W2));
  std::string FetchErr2 = testing::internal::GetCapturedStderr();
  EXPECT_EQ(W2.FlatsWritten, 1);
  EXPECT_EQ(FetchErr2.find("would have waited for a lock"), std::string::npos);
}

TEST_F(ContextTest, FlatProfileSkipsWhenArenaMutexLocked) {
  __llvm_ctx_profile_start_collection();
  FunctionData FData;
  int FakeCalleeAddress = 0;
  {
    __sanitizer::GenericScopedLock<__sanitizer::SpinMutex> Lock(
        &FlatCtxArenaMutex);
    auto *Ctx =
        __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
    EXPECT_TRUE(isScratch(Ctx));
    EXPECT_EQ(FData.FlatCtx, nullptr);
    // Failing to lock FlatCtxArenaMutex must release FData.Mutex.
    if (FData.Mutex.TryLock())
      FData.Mutex.Unlock();
    else
      ADD_FAILURE() << "FData.Mutex remained locked";
  }

  FlatCounterWriter W1;
  testing::internal::CaptureStderr();
  EXPECT_TRUE(__llvm_ctx_profile_fetch(W1));
  std::string FetchErr1 = testing::internal::GetCapturedStderr();
  EXPECT_EQ(W1.FlatsWritten, 0);
  EXPECT_NE(FetchErr1.find("would have waited for a lock: 1\n"),
            std::string::npos);

  __llvm_ctx_profile_start_collection();
  auto *Ctx =
      __llvm_ctx_profile_get_context(&FData, &FakeCalleeAddress, 2, 3, 1);
  EXPECT_TRUE(isScratch(Ctx));
  ASSERT_NE(FData.FlatCtx, nullptr);
  EXPECT_EQ(FData.FlatCtx->entrycount(), 1U);

  FlatCounterWriter W2;
  testing::internal::CaptureStderr();
  EXPECT_TRUE(__llvm_ctx_profile_fetch(W2));
  std::string FetchErr2 = testing::internal::GetCapturedStderr();
  EXPECT_EQ(W2.FlatsWritten, 1);
  EXPECT_EQ(FetchErr2.find("would have waited for a lock"), std::string::npos);
}

TEST_F(ContextTest, FlatProfileConcurrentAllocation) {
  __llvm_ctx_profile_start_collection();
  constexpr int NumThreads = 4;
  constexpr int NumFuncs = 16;
  FunctionData SharedFuncs[NumFuncs];
  int FakeAddrs[NumFuncs] = {};

  std::thread Threads[NumThreads];
  for (int T = 0; T < NumThreads; ++T) {
    Threads[T] = std::thread([&]() {
      for (int Iter = 0; Iter < 8; ++Iter) {
        for (int F = 0; F < NumFuncs; ++F) {
          auto *Ctx = __llvm_ctx_profile_get_context(
              &SharedFuncs[F], &FakeAddrs[F], 100 + F, 2, 0);
          EXPECT_TRUE(isScratch(Ctx));
        }
      }
    });
  }
  for (auto &T : Threads)
    T.join();

  // Verify no function mutex was left locked after concurrent misses, and
  // allocate any function whose initial attempts were skipped under contention.
  for (int F = 0; F < NumFuncs; ++F) {
    if (SharedFuncs[F].Mutex.TryLock())
      SharedFuncs[F].Mutex.Unlock();
    else
      ADD_FAILURE() << "SharedFuncs[" << F << "].Mutex remained locked";
    auto *Ctx = __llvm_ctx_profile_get_context(&SharedFuncs[F], &FakeAddrs[F],
                                               100 + F, 2, 0);
    EXPECT_TRUE(isScratch(Ctx));
    ASSERT_NE(SharedFuncs[F].FlatCtx, nullptr);
  }

  FlatCounterWriter W;
  EXPECT_TRUE(__llvm_ctx_profile_fetch(W));
  EXPECT_EQ(W.FlatsWritten, NumFuncs);
  for (const auto &Entry : W.FlatProfiles) {
    EXPECT_EQ(Entry.second.size(), 2U);
    EXPECT_GE(Entry.second[0], 1U);
  }
}

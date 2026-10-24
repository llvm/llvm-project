//===-- copyprof_state_test.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "copyprof/copyprof_state.h"

#include <errno.h>
#include <pthread.h>
#include <string.h>

#include "copyprof/copyprof_allocator.h"
#include "copyprof/copyprof_interface_internal.h"
#include "copyprof/copyprof_per_object_state.h"
#include "copyprof/copyprof_shadow.h"
#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_allocator.h"

namespace __copyprof {
namespace {

// The runtime keeps its per-thread state in a single object that the callbacks
// mutate in place, so without a reset each test would inherit whatever nesting
// levels, mode and flags the previous one left behind -- and a test that
// aborts mid-scenario would cascade into the rest of the suite.
//
// This does NOT reset shadow memory: it is process-global and far too large to
// clear between tests, and the buffers under test are stack addresses that get
// reused across cases. Tests that depend on the shadow state of a buffer must
// establish it explicitly with `MarkApplicationMemory`.
class CopyProfStateTest : public testing::Test {
 protected:
  void SetUp() override {
    __copyprof_init_once();
    __copyprof_state = PerThreadState();
    InitializePerObjectTracking();
  }
};

TEST_F(CopyProfStateTest, NestingCounters) {
  EXPECT_EQ(__copyprof_state.construct_nesting_level, 0u);
  EXPECT_EQ(__copyprof_state.copy_nesting_level, 0u);
  EXPECT_EQ(__copyprof_state.destruct_nesting_level, 0u);
  EXPECT_EQ(__copyprof_state.smf_context, SmfContext::NONE);

  // Enter constructor callback.
  __copyprof_ctor_enter_callback(/*this_ptr=*/nullptr, /*obj_size=*/16);
  EXPECT_EQ(__copyprof_state.construct_nesting_level, 1u);
  EXPECT_EQ(__copyprof_state.smf_context, SmfContext::CTOR);

  // Exit constructor callback.
  __copyprof_ctor_exit_callback(/*this_ptr=*/nullptr, /*obj_size=*/16);
  EXPECT_EQ(__copyprof_state.construct_nesting_level, 0u);
  EXPECT_EQ(__copyprof_state.smf_context, SmfContext::NONE);
}

TEST_F(CopyProfStateTest, SimulatedUnnecessaryCopy) {
  char obj[32] = {0};
  char other[32] = {0};

  // Simulate copy constructor execution.
  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  EXPECT_EQ(__copyprof_state.copy_nesting_level, 1u);
  EXPECT_EQ(__copyprof_state.smf_context, SmfContext::COPY);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  EXPECT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));
  EXPECT_EQ(__copyprof_state.smf_context, SmfContext::NONE);

  // Simulate destructor without any intervening store callbacks.
  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  EXPECT_EQ(__copyprof_state.smf_context, SmfContext::DTOR);
  EXPECT_TRUE(__copyprof_state.is_transitive_copy);
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
}

TEST_F(CopyProfStateTest, ModifiedCopyIsNotMakedAsCopyAnymore) {
  char obj[32] = {0};
  char other[32] = {0};

  // Create copy `obj` based on `other`.
  __copyprof_copy_assign_op_enter_callback(obj, other, sizeof(obj));
  __copyprof_copy_assign_op_exit_callback(obj, other, sizeof(obj));
  EXPECT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));

  // Modifying `obj` must mark it as non-copy.
  __copyprof_store_callback(obj + 4, /*size=*/4);
  EXPECT_FALSE(IsMarkedAsCopy(obj, sizeof(obj)));

  // After destroying the modified copy, `is_transitive_copy` must be `false`.
  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_FALSE(__copyprof_state.is_transitive_copy);
}

TEST_F(CopyProfStateTest, CheckModeIgnoresStores) {
  char obj[32] = {0};
  char other[32] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));
  EXPECT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));

  __copyprof_dtor_enter_callback(obj, sizeof(obj));

  // Simulate an internal store during destruction (e.g. vtable rewrite or
  // member cleanup).
  __copyprof_store_callback(obj, sizeof(obj));

  // Prove that stores in CHECK mode do not clear the copy shadow bits.
  EXPECT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
}

// Unlike a c'tor, a copy assignment inside a d'tor overwrites an existing
// object, so it must leave shadow memory alone, just like a store.
TEST_F(CopyProfStateTest, CheckModeIgnoresCopyAssignment) {
  char obj[32] = {0};
  char other[32] = {0};
  char target[32] = {0};

  // Mark `target` as non-copy up front so a stray shadow write is observable.
  MarkApplicationMemory(target, sizeof(target), /*is_copy=*/false);

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  // Enter the d'tor, then copy-assign to `target`.
  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  __copyprof_copy_assign_op_enter_callback(target, other, sizeof(target));
  __copyprof_copy_assign_op_exit_callback(target, other, sizeof(target));
  EXPECT_FALSE(IsMarkedAsCopy(target, sizeof(target)));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
}

// A d'tor whose body constructs and destroys an object that the destroyed
// object does not own (a string built for a log message, a lock guard, an
// iterator) must not suppress the report. Objects constructed inside a d'tor
// are marked as copy, so destroying them leaves `is_transitive_copy` unchanged.
// Contrast with SimulatedUnnecessaryCopy, which is the same scenario with an
// empty d'tor body.
TEST_F(CopyProfStateTest, TemporaryObjectInDtorDoesNotSuppressReport) {
  char obj[32] = {0};
  char source_obj[32] = {0};
  char local[32] = {0};

  // `obj` is an unnecessary copy: copy-constructed and never modified.
  __copyprof_copy_ctor_enter_callback(obj, source_obj, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, source_obj, sizeof(obj));
  ASSERT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));

  // Memory occupied by `local` is not a copy.
  MarkApplicationMemory(local, sizeof(local), /*is_copy=*/false);

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  // Create and destroy `local` while `obj`'s d'tor is running.
  __copyprof_ctor_enter_callback(local, sizeof(local));
  __copyprof_ctor_exit_callback(local, sizeof(local));
  __copyprof_dtor_enter_callback(local, sizeof(local));
  __copyprof_dtor_exit_callback(local, sizeof(local));

  // `obj` itself was never modified, so it must still be marked as copy.
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_TRUE(IsMarkedAsCopy(obj, sizeof(obj)));
  EXPECT_TRUE(__copyprof_state.is_transitive_copy);
}

// Likewise for a temporary that the d'tor copy-constructs, e.g. from a member.
TEST_F(CopyProfStateTest, CopiedTemporaryInDtorDoesNotSuppressReport) {
  char obj[32] = {0};
  char source_obj[32] = {0};
  char temp[32] = {0};

  __copyprof_copy_ctor_enter_callback(obj, source_obj, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, source_obj, sizeof(obj));
  // Memory occupied by `temp` is not a copy.
  MarkApplicationMemory(temp, sizeof(temp), /*is_copy=*/false);

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  __copyprof_copy_ctor_enter_callback(temp, obj, sizeof(temp));
  __copyprof_copy_ctor_exit_callback(temp, obj, sizeof(temp));
  __copyprof_dtor_enter_callback(temp, sizeof(temp));
  __copyprof_dtor_exit_callback(temp, sizeof(temp));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_TRUE(__copyprof_state.is_transitive_copy);
}

// A temporary with a trivial c'tor gets no c'tor callback, so it is not marked
// as copy when a d'tor creates it. If its d'tor is instrumented, destroying it
// folds whatever shadow its memory had before into `is_transitive_copy`, which
// may suppress the report.
//
// DISABLED: known limitation. Possible fixes are to also instrument trivial
// construction, or to skip the fold for objects in the stack frames below the
// top-level d'tor.
TEST_F(CopyProfStateTest,
       DISABLED_TriviallyConstructedTemporaryInDtorDoesNotSuppressReport) {
  char obj[32] = {0};
  char source_obj[32] = {0};
  char temp[32] = {0};

  __copyprof_copy_ctor_enter_callback(obj, source_obj, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, source_obj, sizeof(obj));
  // Memory occupied by `temp` is not a copy.
  MarkApplicationMemory(temp, sizeof(temp), /*is_copy=*/false);

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  // `temp` is created without a c'tor callback, then destroyed.
  __copyprof_dtor_enter_callback(temp, sizeof(temp));
  __copyprof_dtor_exit_callback(temp, sizeof(temp));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_TRUE(__copyprof_state.is_transitive_copy);
}

// Likewise for a temporary that a d'tor move-constructs, e.g. by moving a
// member out (`std::string tmp = std::move(name_);`). Move c'tors are not
// instrumented, so there is no c'tor callback. The same happens for c'tors from
// an uninstrumented library, e.g. `std::string`'s copy c'tor from libstdc++ at
// -O2 in C++17 mode.
//
// DISABLED: known limitation, see
// TriviallyConstructedTemporaryInDtorDoesNotSuppressReport.
TEST_F(CopyProfStateTest, DISABLED_MovedTemporaryInDtorDoesNotSuppressReport) {
  char obj[32] = {0};
  char source_obj[32] = {0};
  char temp[32] = {0};

  // `obj` is a copy that owns a buffer allocated by its copy c'tor.
  __copyprof_copy_ctor_enter_callback(obj, source_obj, sizeof(obj));
  void* buf = CopyProfMalloc(/*size=*/64);
  __copyprof_copy_ctor_exit_callback(obj, source_obj, sizeof(obj));
  ASSERT_NE(buf, nullptr);
  // Memory occupied by `temp` is not a copy.
  MarkApplicationMemory(temp, sizeof(temp), /*is_copy=*/false);

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  // Move `buf` from `obj` into `temp`. The move c'tor's stores are ignored in
  // `DTOR` context, and there is no c'tor callback.
  __copyprof_store_callback(temp, sizeof(temp));
  __copyprof_store_callback(obj, sizeof(obj));
  // Destroying `temp` frees `buf`.
  __copyprof_dtor_enter_callback(temp, sizeof(temp));
  CopyProfFree(buf);
  __copyprof_dtor_exit_callback(temp, sizeof(temp));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_TRUE(__copyprof_state.is_transitive_copy);
}

TEST_F(CopyProfStateTest, PerObjectStateTrackingWorksCorrectly) {
  char obj[24] = {0};

  // Querying an untracked object returns 0 / false.
  EXPECT_EQ(GetPerObjectTrackedSize(obj), 0);
  EXPECT_FALSE(GetPerObjectDidAllocate(obj));

  InitPerObjectState(obj, sizeof(obj));
  EXPECT_EQ(GetPerObjectTrackedSize(obj), sizeof(obj));
  EXPECT_FALSE(GetPerObjectDidAllocate(obj));

  RecordPerObjectAllocation(obj, /*num_bytes=*/77);
  EXPECT_EQ(GetPerObjectTrackedSize(obj), sizeof(obj) + 77);
  EXPECT_TRUE(GetPerObjectDidAllocate(obj));

  RecordPerObjectAllocation(obj, /*num_bytes=*/32);
  EXPECT_EQ(GetPerObjectTrackedSize(obj), sizeof(obj) + 77 + 32);

  RemovePerObjectState(obj);
  EXPECT_EQ(GetPerObjectTrackedSize(obj), 0);
  EXPECT_FALSE(GetPerObjectDidAllocate(obj));
}

// Only a top-level copy gets per-object state, which is removed again when the
// top-level d'tor exits. A copy nested in a c'tor (e.g. a member copied from a
// c'tor argument) is eventually destroyed by a nested d'tor, so state created
// for it would never be removed.
TEST_F(CopyProfStateTest, CopyNestedInConstructorHasNoPerObjectState) {
  char obj[32] = {0};
  char other[16] = {0};
  char* member = obj + 16;

  __copyprof_ctor_enter_callback(obj, sizeof(obj));
  __copyprof_copy_ctor_enter_callback(member, other, sizeof(other));
  __copyprof_copy_ctor_exit_callback(member, other, sizeof(other));
  __copyprof_ctor_exit_callback(obj, sizeof(obj));

  EXPECT_EQ(GetPerObjectTrackedSize(member), 0u);
}

// Likewise for a copy nested in a d'tor, e.g. a temporary created by its body.
TEST_F(CopyProfStateTest, CopyNestedInDestructorHasNoPerObjectState) {
  char obj[16] = {0};
  char temp[16] = {0};
  char other[16] = {0};

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  __copyprof_copy_ctor_enter_callback(temp, other, sizeof(temp));
  __copyprof_copy_ctor_exit_callback(temp, other, sizeof(temp));
  __copyprof_dtor_enter_callback(temp, sizeof(temp));
  __copyprof_dtor_exit_callback(temp, sizeof(temp));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));

  EXPECT_EQ(GetPerObjectTrackedSize(temp), 0u);
}

TEST_F(CopyProfStateTest, AllocationDuringCopyIsRecordedInPerObjectState) {
  char obj[16] = {0};
  char other[16] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  void* buf = CopyProfAllocate(/*size=*/32, /*align=*/8);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(GetPerObjectTrackedSize(obj), sizeof(obj) + 32);
  EXPECT_TRUE(GetPerObjectDidAllocate(obj));
  CopyProfDeallocate(buf, /*size=*/32, /*align=*/8);

  // Destroying `obj` must erase its object state and hence the tracked size
  // must be zero.
  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_EQ(GetPerObjectTrackedSize(obj), 0);
}

TEST_F(CopyProfStateTest, FailedAllocationDuringCopyIsNotRecorded) {
  char obj[16] = {0};
  char other[16] = {0};
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  // A non-power-of-two alignment makes the allocation fail.
  void* buf = CopyProfAllocate(/*size=*/32, /*align=*/3);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));
  SetAllocatorMayReturnNull(may_return_null);

  EXPECT_EQ(buf, nullptr);
  EXPECT_EQ(GetPerObjectTrackedSize(obj), sizeof(obj));
  EXPECT_FALSE(GetPerObjectDidAllocate(obj));

  // Destroying `obj` must erase its object state and hence the tracked size
  // must be zero.
  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
  EXPECT_EQ(GetPerObjectTrackedSize(obj), 0);
}

TEST_F(CopyProfStateTest,
       DeallocatingUnmodifiedCopyAllocationPreservesTransitiveCopy) {
  char obj[16] = {0};
  char other[16] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  void* copy_buf = CopyProfAllocate(/*size=*/32, /*align=*/8);
  ASSERT_NE(copy_buf, nullptr);
  EXPECT_TRUE(IsMarkedAsCopy(copy_buf, /*num_bytes=*/32));
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  CopyProfDeallocate(copy_buf, /*size=*/32, /*align=*/8);
  EXPECT_TRUE(__copyprof_state.is_transitive_copy);
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
}

TEST_F(CopyProfStateTest,
       DeallocatingModifiedCopyAllocationClearsTransitiveCopy) {
  char obj[16] = {0};
  char other[16] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  void* modified_buf = CopyProfAllocate(/*size=*/32, /*align=*/8);
  ASSERT_NE(modified_buf, nullptr);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  __copyprof_store_callback(static_cast<char*>(modified_buf) + 8, /*size=*/1);
  EXPECT_FALSE(IsMarkedAsCopy(modified_buf, /*num_bytes=*/32));

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  CopyProfDeallocate(modified_buf, /*size=*/32, /*align=*/8);
  EXPECT_FALSE(__copyprof_state.is_transitive_copy);
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
}

TEST_F(CopyProfStateTest, ZeroSizedAllocation) {
  void* zero_buf = CopyProfAllocate(/*size=*/0, /*align=*/8);
  ASSERT_NE(zero_buf, nullptr);
  CopyProfDeallocate(zero_buf, /*size=*/0, /*align=*/8);
}

TEST_F(CopyProfStateTest, ThreadLocalAllocatorCacheAndDrainOnExit) {
  auto worker = +[](void* /*arg*/) -> void* {
    void* local_buf = CopyProfAllocate(/*size=*/64, /*align=*/8);
    CopyProfDeallocate(local_buf, /*size=*/64, /*align=*/8);
    return CopyProfAllocate(/*size=*/128, /*align=*/8);
  };
  pthread_t thread;
  ASSERT_EQ(pthread_create(&thread, /*attr=*/nullptr, worker, /*arg=*/nullptr),
            0);
  void* cross_thread_buf = nullptr;
  ASSERT_EQ(pthread_join(thread, &cross_thread_buf), 0);
  ASSERT_NE(cross_thread_buf, nullptr);
  CopyProfDeallocate(cross_thread_buf, /*size=*/128, /*align=*/8);
}

// The exiting thread may still (de)allocate after its cache has been committed
// back, e.g. in TSD destructors that run later.
TEST_F(CopyProfStateTest, AllocationAfterThreadCacheCommitBackSucceeds) {
  static pthread_key_t key;
  static void* late_buf;
  late_buf = nullptr;
  auto destructor = +[](void* value) {
    if (value == &key) {
      // Invoking `pthread_setspecific` forces `pthread_exit` to perform another
      // pass over TSD destructors, resulting in execution of the code path
      // below afterwards.
      pthread_setspecific(key, &late_buf);
      return;
    }
    late_buf = CopyProfMalloc(/*size=*/16);
    CopyProfFree(late_buf);
  };
  ASSERT_EQ(pthread_key_create(&key, destructor), 0);
  auto worker = +[](void* /*arg*/) -> void* {
    // Allocate once to create the thread cache and arm its TSD destructor.
    CopyProfFree(CopyProfMalloc(/*size=*/16));
    pthread_setspecific(key, &key);
    return nullptr;
  };
  pthread_t thread;
  ASSERT_EQ(pthread_create(&thread, /*attr=*/nullptr, worker, /*arg=*/nullptr),
            0);
  ASSERT_EQ(pthread_join(thread, /*thread_return=*/nullptr), 0);
  pthread_key_delete(key);
  EXPECT_NE(late_buf, nullptr);
}

TEST_F(CopyProfStateTest, MallocDuringCopyIsMarkedAsCopy) {
  char obj[16] = {0};
  char other[16] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  void* buf = CopyProfMalloc(/*size=*/32);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  ASSERT_NE(buf, nullptr);
  EXPECT_TRUE(IsMarkedAsCopy(buf, /*num_bytes=*/32));
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, FreeingModifiedCopyAllocationClearsTransitiveCopy) {
  char obj[16] = {0};
  char other[16] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  void* buf = CopyProfMalloc(/*size=*/32);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));
  ASSERT_NE(buf, nullptr);
  // Modifying `buf` marks it as non-copy.
  __copyprof_store_callback(buf, /*size=*/1);

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  CopyProfFree(buf);
  EXPECT_FALSE(__copyprof_state.is_transitive_copy);
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
}

// Freeing a temporary allocated by a d'tor must not clear `is_transitive_copy`,
// regardless of the shadow state that the chunk's previous owner left behind.
TEST_F(CopyProfStateTest, TemporaryAllocationInDtorPreservesTransitiveCopy) {
  char obj[16] = {0};
  char other[16] = {0};
  // Free a non-copy chunk so that the temporary below likely reuses it.
  CopyProfFree(CopyProfMalloc(/*size=*/32));

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  __copyprof_dtor_enter_callback(obj, sizeof(obj));
  void* temp = CopyProfMalloc(/*size=*/32);
  ASSERT_NE(temp, nullptr);
  CopyProfFree(temp);
  EXPECT_TRUE(__copyprof_state.is_transitive_copy);
  __copyprof_dtor_exit_callback(obj, sizeof(obj));
}

// A d'tor (here `donor`'s) may allocate memory and hand it to another live copy
// (here `holder`). The store into `holder` happens in `DTOR` context and is
// ignored, and the memory is marked as copy, so `holder` looks unmodified when
// it is destroyed and is reported as an unnecessary copy.
//
// DISABLED: accepted false positive. Marking memory allocated in `DTOR` context
// as copy keeps temporaries from suppressing reports (see
// TemporaryAllocationInDtorPreservesTransitiveCopy), which matters far more in
// practice.
TEST_F(CopyProfStateTest, DISABLED_CopyModifiedByAnotherDtorIsNotReported) {
  char holder[32] = {0};
  char source_holder[32] = {0};
  char donor[16] = {0};

  __copyprof_copy_ctor_enter_callback(holder, source_holder, sizeof(holder));
  __copyprof_copy_ctor_exit_callback(holder, source_holder, sizeof(holder));

  // `donor`'s d'tor allocates `extra` and stores a pointer to it in `holder`.
  __copyprof_dtor_enter_callback(donor, sizeof(donor));
  void* extra = CopyProfMalloc(/*size=*/16);
  ASSERT_NE(extra, nullptr);
  __copyprof_store_callback(holder, /*size=*/sizeof(extra));
  __copyprof_dtor_exit_callback(donor, sizeof(donor));

  // `holder`'s d'tor frees `extra`.
  __copyprof_dtor_enter_callback(holder, sizeof(holder));
  CopyProfFree(extra);
  __copyprof_dtor_exit_callback(holder, sizeof(holder));
  EXPECT_FALSE(__copyprof_state.is_transitive_copy);
}

TEST_F(CopyProfStateTest, CallocReturnsZeroedMemory) {
  constexpr uptr kSize = 32;
  // Dirty a chunk and free it again, s.t. `CopyProfCalloc` likely reuses it.
  void* dirty_buf = CopyProfMalloc(kSize);
  ASSERT_NE(dirty_buf, nullptr);
  memset(dirty_buf, /*c=*/0xab, kSize);
  CopyProfFree(dirty_buf);

  char* buf = static_cast<char*>(CopyProfCalloc(/*num=*/1, kSize));
  ASSERT_NE(buf, nullptr);
  for (uptr i = 0; i < kSize; ++i) EXPECT_EQ(buf[i], 0) << "at offset " << i;
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, CallocDuringCopyIsMarkedAsCopy) {
  char obj[16] = {0};
  char other[16] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  void* buf = CopyProfCalloc(/*num=*/4, /*size=*/8);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  ASSERT_NE(buf, nullptr);
  EXPECT_TRUE(IsMarkedAsCopy(buf, /*num_bytes=*/32));
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, CallocOverflowReturnsNull) {
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);
  errno = 0;

  // `num * size` overflows.
  void* buf = CopyProfCalloc(/*num=*/2, /*size=*/~uptr{0});
  const int calloc_errno = errno;
  SetAllocatorMayReturnNull(may_return_null);

  EXPECT_EQ(buf, nullptr);
  EXPECT_EQ(calloc_errno, ENOMEM);
}

TEST_F(CopyProfStateTest, ReallocPreservesContents) {
  constexpr uptr kSize = 16;
  char* buf = static_cast<char*>(CopyProfMalloc(kSize));
  ASSERT_NE(buf, nullptr);
  for (uptr i = 0; i < kSize; ++i) buf[i] = static_cast<char>(i);

  char* resized_buf =
      static_cast<char*>(CopyProfRealloc(buf, /*size=*/2 * kSize));
  ASSERT_NE(resized_buf, nullptr);
  for (uptr i = 0; i < kSize; ++i)
    EXPECT_EQ(resized_buf[i], static_cast<char>(i)) << "at offset " << i;
  CopyProfFree(resized_buf);
}

TEST_F(CopyProfStateTest, ReallocOfNullptrAllocates) {
  void* buf = CopyProfRealloc(/*ptr=*/nullptr, /*size=*/32);
  EXPECT_NE(buf, nullptr);
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, ReallocToZeroSizeReturnsNull) {
  void* buf = CopyProfMalloc(/*size=*/32);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(CopyProfRealloc(buf, /*size=*/0), nullptr);
}

TEST_F(CopyProfStateTest, ReallocDuringCopyIsMarkedAsCopy) {
  char obj[16] = {0};
  char other[16] = {0};
  void* buf = CopyProfMalloc(/*size=*/16);
  ASSERT_NE(buf, nullptr);

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  void* resized_buf = CopyProfRealloc(buf, /*size=*/32);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));

  ASSERT_NE(resized_buf, nullptr);
  EXPECT_TRUE(IsMarkedAsCopy(resized_buf, /*num_bytes=*/32));
  CopyProfFree(resized_buf);
}

// The shadow of reallocated memory follows the SMF context of the `realloc`
// call; the shadow of the original memory is not carried over.
TEST_F(CopyProfStateTest, ReallocOutsideCopyIsNotMarkedAsCopy) {
  char obj[16] = {0};
  char other[16] = {0};

  __copyprof_copy_ctor_enter_callback(obj, other, sizeof(obj));
  void* buf = CopyProfMalloc(/*size=*/16);
  __copyprof_copy_ctor_exit_callback(obj, other, sizeof(obj));
  ASSERT_NE(buf, nullptr);
  ASSERT_TRUE(IsMarkedAsCopy(buf, /*num_bytes=*/16));

  void* resized_buf = CopyProfRealloc(buf, /*size=*/32);
  ASSERT_NE(resized_buf, nullptr);
  EXPECT_FALSE(IsMarkedAsCopy(resized_buf, /*num_bytes=*/32));
  CopyProfFree(resized_buf);
}

TEST_F(CopyProfStateTest, AllocateWithInvalidAlignmentSetsEinval) {
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);

  for (uptr align : {uptr{0}, uptr{3}}) {
    errno = 0;
    void* buf = CopyProfAllocate(/*size=*/32, align);
    EXPECT_EQ(buf, nullptr) << "for align " << align;
    EXPECT_EQ(errno, EINVAL) << "for align " << align;
  }

  SetAllocatorMayReturnNull(may_return_null);
}

TEST_F(CopyProfStateTest, AllocateReturnsMemoryWithRequestedAlignment) {
  constexpr uptr kAlignment = 256;
  void* buf = CopyProfAllocate(/*size=*/16, kAlignment);
  ASSERT_NE(buf, nullptr);
  EXPECT_TRUE(IsAligned(reinterpret_cast<uptr>(buf), kAlignment));
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, PosixMemalignReturnsMemoryWithRequestedAlignment) {
  constexpr uptr kAlignment = 64;
  void* buf = nullptr;
  ASSERT_EQ(CopyProfPosixMemalign(&buf, kAlignment, /*size=*/100), 0);
  EXPECT_TRUE(IsAligned(reinterpret_cast<uptr>(buf), kAlignment));
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, PosixMemalignWithInvalidAlignmentReturnsEinval) {
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);
  void* buf = nullptr;

  // Not a multiple of `sizeof(void*)`.
  const int result = CopyProfPosixMemalign(&buf, /*alignment=*/4, /*size=*/16);
  SetAllocatorMayReturnNull(may_return_null);

  EXPECT_EQ(result, EINVAL);
  EXPECT_EQ(buf, nullptr);
}

TEST_F(CopyProfStateTest, VallocReturnsPageAlignedMemory) {
  void* buf = CopyProfValloc(/*size=*/16);
  ASSERT_NE(buf, nullptr);
  EXPECT_TRUE(IsAligned(reinterpret_cast<uptr>(buf), GetPageSizeCached()));
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, PvallocRoundsSizeUpToPageSize) {
  void* buf = CopyProfPvalloc(/*size=*/1);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(CopyProfMallocUsableSize(buf), GetPageSizeCached());
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, ReallocarrayAllocatesNumTimesSize) {
  void* buf = CopyProfReallocarray(/*ptr=*/nullptr, /*num=*/4, /*size=*/8);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(CopyProfMallocUsableSize(buf), 32u);
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, ReallocarrayOverflowReturnsNull) {
  const bool may_return_null = AllocatorMayReturnNull();
  SetAllocatorMayReturnNull(true);
  errno = 0;

  // `num * size` overflows.
  void* buf =
      CopyProfReallocarray(/*ptr=*/nullptr, /*num=*/2, /*size=*/~uptr{0});
  const int reallocarray_errno = errno;
  SetAllocatorMayReturnNull(may_return_null);

  EXPECT_EQ(buf, nullptr);
  EXPECT_EQ(reallocarray_errno, ENOMEM);
}

TEST_F(CopyProfStateTest, MallocUsableSizeReturnsRequestedSize) {
  void* buf = CopyProfMalloc(/*size=*/10);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(CopyProfMallocUsableSize(buf), 10u);
  CopyProfFree(buf);
}

TEST_F(CopyProfStateTest, MallocUsableSizeOfUnownedMemoryIsZero) {
  char stack_buf[16] = {0};
  EXPECT_EQ(CopyProfMallocUsableSize(stack_buf), 0u);
}

}  // namespace
}  // namespace __copyprof

//===-- copyprof_state_test.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "copyprof/copyprof_state.h"

#include "copyprof/copyprof_interface_internal.h"
#include "copyprof/copyprof_shadow.h"
#include "gtest/gtest.h"

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

}  // namespace
}  // namespace __copyprof

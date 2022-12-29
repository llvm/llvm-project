// RUN: %clang_dfsan -g3 -DDATA_BYTES=3 %s -fno-exceptions -latomic -o %t && %run %t
// RUN: %clang_dfsan -g3 -DDATA_BYTES=3 -DORIGIN_TRACKING -mllvm -dfsan-track-origins=1 %s -fno-exceptions -latomic -o %t && %run %t
// RUN: %clang_dfsan -g3 -DDATA_BYTES=32 %s -fno-exceptions -latomic -o %t && %run %t
// RUN: %clang_dfsan -g3 -DDATA_BYTES=32 -DORIGIN_TRACKING -mllvm -dfsan-track-origins=1 %s -fno-exceptions -latomic -o %t && %run %t

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <stdatomic.h>

typedef struct __attribute((packed)) {
  uint8_t val[DATA_BYTES];
} idata;

void test_idata_load() {
  idata dest = {-1};
  idata init = {0};

  dfsan_label i_label = 2;
  dfsan_set_label(i_label, &init, sizeof(init));

  __atomic_load(&init, &dest, __ATOMIC_RELAXED);

  dfsan_label read_label = dfsan_read_label(&dest, sizeof(dest));
  assert(read_label == i_label);
#ifdef ORIGIN_TRACKING
  dfsan_origin read_origin =
      dfsan_read_origin_of_first_taint(&dest, sizeof(dest));
  assert(read_origin != 0);
#endif
}

void test_idata_store() {
  idata dest = {-1};
  idata init = {0};

  dfsan_label i_label = 2;
  dfsan_set_label(i_label, &init, sizeof(init));

  __atomic_store(&init, &dest, __ATOMIC_RELAXED);

  dfsan_label read_label = dfsan_read_label(&dest, sizeof(dest));
  assert(read_label == i_label);
#ifdef ORIGIN_TRACKING
  dfsan_origin read_origin =
      dfsan_read_origin_of_first_taint(&dest, sizeof(dest));
  assert(read_origin != 0);
#endif
}

void test_idata_exchange() {
  idata target = {-1};
  idata init = {0};
  idata dest = {3};

  dfsan_label i_label = 1;
  dfsan_set_label(i_label, &init, sizeof(init));
  dfsan_label j_label = 2;
  dfsan_set_label(j_label, &target, sizeof(target));

  dfsan_label dest0_label = dfsan_read_label(&dest, sizeof(dest));
  assert(dest0_label == 0);
#ifdef ORIGIN_TRACKING
  dfsan_origin dest0_origin =
      dfsan_read_origin_of_first_taint(&dest, sizeof(dest));
  assert(dest0_origin == 0);
#endif

  __atomic_exchange(&target, &init, &dest, __ATOMIC_RELAXED);

  dfsan_label dest_label = dfsan_read_label(&dest, sizeof(dest));
  assert(dest_label == j_label);
#ifdef ORIGIN_TRACKING
  dfsan_origin dest_origin =
      dfsan_read_origin_of_first_taint(&dest, sizeof(dest));
  assert(dest_origin != 0);
#endif

  dfsan_label target_label = dfsan_read_label(&target, sizeof(target));
  assert(target_label == i_label);
#ifdef ORIGIN_TRACKING
  dfsan_origin target_origin =
      dfsan_read_origin_of_first_taint(&target, sizeof(target));
  assert(target_origin != 0);
#endif
}

void test_idata_cmp_exchange_1() {
  idata target = {0};
  idata expected = {0}; // Target matches expected
  idata desired = {3};

  dfsan_label i_label = 1;
  dfsan_set_label(i_label, &expected, sizeof(expected));
  dfsan_label j_label = 2;
  dfsan_set_label(j_label, &target, sizeof(target));
  dfsan_label k_label = 4;
  dfsan_set_label(k_label, &desired, sizeof(desired));

  int r =
      __atomic_compare_exchange(&target, &expected, &desired, /*weak=false*/ 0,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  // Target matches expected => true
  assert(r);

  // Copy desired to target.
  dfsan_label target_label = dfsan_read_label(&target, sizeof(target));
  assert(target_label == k_label);
#ifdef ORIGIN_TRACKING
  dfsan_origin target_origin =
      dfsan_read_origin_of_first_taint(&target, sizeof(target));
  assert(target_origin != 0);
#endif
}

void test_idata_cmp_exchange_2() {
  idata target = {0};
  idata expected = {-1}; // Target does not match expected
  idata desired = {3};

  dfsan_label i_label = 1;
  dfsan_set_label(i_label, &expected, sizeof(expected));
  dfsan_label j_label = 2;
  dfsan_set_label(j_label, &target, sizeof(target));
  dfsan_label k_label = 4;
  dfsan_set_label(k_label, &desired, sizeof(desired));

  int r =
      __atomic_compare_exchange(&target, &expected, &desired, /*weak=false*/ 0,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  // Target does not match expected => false
  assert(!r);

  // Copy target to expected
  dfsan_label expected_label = dfsan_read_label(&expected, sizeof(expected));
  assert(expected_label == j_label);
#ifdef ORIGIN_TRACKING
  dfsan_origin expected_origin =
      dfsan_read_origin_of_first_taint(&expected, sizeof(expected));
  assert(expected_origin != 0);
#endif
}

int main() {
  test_idata_load();
  test_idata_store();
  test_idata_exchange();
  test_idata_cmp_exchange_1();
  test_idata_cmp_exchange_2();

  return 0;
}

// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu zEC12 -triple s390x-ibm-linux -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu zEC12 -triple s390x-ibm-linux -Wall -Wno-unused -Werror -emit-llvm -x c++ %s -o - | FileCheck %s

#include <stdint.h>
#include <htmintrin.h>

int global = 0;
uint64_t g;
struct __htm_tdb global_tdb;

void test_htm1(struct __htm_tdb *tdb, int reg, int *mem, uint64_t *mem64) {
// CHECK-LABEL: test_htm1

  __builtin_tbegin ((void *)0);
// CHECK: call i32 @llvm.s390.tbegin(ptr null, i32 65292)
  __builtin_tbegin ((void *)0x12345678);
// CHECK: call i32 @llvm.s390.tbegin(ptr inttoptr (i64 305419896 to ptr), i32 65292)
  __builtin_tbegin (tdb);
// CHECK: call i32 @llvm.s390.tbegin(ptr %{{.*}}, i32 65292)
  __builtin_tbegin (&global_tdb);
// CHECK: call i32 @llvm.s390.tbegin(ptr @global_tdb, i32 65292)

  __builtin_tbegin_nofloat ((void *)0);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr null, i32 65292)
  __builtin_tbegin_nofloat ((void *)0x12345678);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr inttoptr (i64 305419896 to ptr), i32 65292)
  __builtin_tbegin_nofloat (tdb);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr %{{.*}}, i32 65292)
  __builtin_tbegin_nofloat (&global_tdb);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr @global_tdb, i32 65292)

  __builtin_tbegin_retry ((void *)0, 6);
// CHECK: call i32 @llvm.s390.tbegin(ptr null, i32 65292)
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
  __builtin_tbegin_retry ((void *)0x12345678, 6);
// CHECK: call i32 @llvm.s390.tbegin(ptr %{{.*}}, i32 65292)
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
  __builtin_tbegin_retry (tdb, 6);
// CHECK: call i32 @llvm.s390.tbegin(ptr %{{.*}}, i32 65292)
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
  __builtin_tbegin_retry (&global_tdb, 6);
// CHECK: call i32 @llvm.s390.tbegin(ptr %{{.*}}, i32 65292)
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})

  __builtin_tbegin_retry_nofloat ((void *)0, 6);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr null, i32 65292)
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
  __builtin_tbegin_retry_nofloat ((void *)0x12345678, 6);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr %{{.*}}, i32 65292)
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
  __builtin_tbegin_retry_nofloat (tdb, 6);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr %{{.*}}, i32 65292)
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
  __builtin_tbegin_retry_nofloat (&global_tdb, 6);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr %{{.*}}, i32 65292)
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})

  __builtin_tbeginc ();
// CHECK: call void @llvm.s390.tbeginc(ptr null, i32 65288)

  __builtin_tabort (256);
// CHECK: call void @llvm.s390.tabort(i64 256)
  __builtin_tabort (-1);
// CHECK: call void @llvm.s390.tabort(i64 -1)
  __builtin_tabort (reg);
// CHECK: call void @llvm.s390.tabort(i64 %{{.*}})

  __builtin_tend();
// CHECK: call i32 @llvm.s390.tend()

  int n = __builtin_tx_nesting_depth();
// CHECK: call i32 @llvm.s390.etnd()

  __builtin_non_tx_store (mem64, 0);
// CHECK: call void @llvm.s390.ntstg(i64 0, ptr %{{.*}})
  const uint64_t val_var = 0x1122334455667788;
  __builtin_non_tx_store (mem64, val_var);
// CHECK: call void @llvm.s390.ntstg(i64 1234605616436508552, ptr %{{.*}})
  __builtin_non_tx_store (mem64, (uint64_t)reg);
// CHECK: call void @llvm.s390.ntstg(i64 %{{.*}}, ptr %{{.*}})
  __builtin_non_tx_store (mem64, g);
// CHECK: call void @llvm.s390.ntstg(i64 %{{.*}}, ptr %{{.*}})
  __builtin_non_tx_store ((uint64_t *)0, 0);
// CHECK: call void @llvm.s390.ntstg(i64 0, ptr null)
  __builtin_non_tx_store ((uint64_t *)0x12345678, 0);
// CHECK: call void @llvm.s390.ntstg(i64 0, ptr inttoptr (i64 305419896 to ptr))
  __builtin_non_tx_store (&g, 23);
// CHECK: call void @llvm.s390.ntstg(i64 23, ptr @g)
  __builtin_non_tx_store (&g, reg);
// CHECK: call void @llvm.s390.ntstg(i64 %{{.*}}, ptr @g)
  __builtin_non_tx_store (&g, *mem);
// CHECK: call void @llvm.s390.ntstg(i64 %{{.*}}, ptr @g)
  __builtin_non_tx_store (&g, global);
// CHECK: call void @llvm.s390.ntstg(i64 %{{.*}}, ptr @g)

  __builtin_tx_assist (0);
// CHECK: call void @llvm.s390.ppa.txassist(i32 0)
  __builtin_tx_assist (1);
// CHECK: call void @llvm.s390.ppa.txassist(i32 1)
  __builtin_tx_assist (reg);
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
  __builtin_tx_assist (*mem);
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
  __builtin_tx_assist (global);
// CHECK: call void @llvm.s390.ppa.txassist(i32 %{{.*}})
}

#include <htmxlintrin.h>

void test_htmxl1(void) {
// CHECK-LABEL: test_htmxl1

  struct __htm_tdb tdb_struct;
  void * const tdb = &tdb_struct;
  long result;
  unsigned char code;

  result = __TM_simple_begin ();
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr null, i32 65292)
  result = __TM_begin (tdb);
// CHECK: call i32 @llvm.s390.tbegin.nofloat(ptr %{{.*}}, i32 65292)
  result = __TM_end ();
// CHECK: call i32 @llvm.s390.tend()
  __TM_abort ();
// CHECK: call void @llvm.s390.tabort(i64 256)
  __TM_named_abort (42);
// CHECK: call void @llvm.s390.tabort(i64 %{{.*}})
  __TM_non_transactional_store (&g, 42);
// CHECK: call void @llvm.s390.ntstg(i64 %{{.*}}, ptr %{{.*}})
  result = __TM_nesting_depth (tdb);
// CHECK: call i32 @llvm.s390.etnd()

  result = __TM_is_user_abort (tdb);
  result = __TM_is_named_user_abort (tdb, &code);
  result = __TM_is_illegal (tdb);
  result = __TM_is_footprint_exceeded (tdb);
  result = __TM_is_nested_too_deep (tdb);
  result = __TM_is_conflict (tdb);
  result = __TM_is_failure_persistent (result);
  result = __TM_failure_address (tdb);
  result = __TM_failure_code (tdb);
}

void test_eh_return_data_regno() {
  volatile int res;
  res = __builtin_eh_return_data_regno(0); // CHECK: store volatile i32 6
  res = __builtin_eh_return_data_regno(1); // CHECK: store volatile i32 7
  res = __builtin_eh_return_data_regno(2); // CHECK: store volatile i32 8
  res = __builtin_eh_return_data_regno(3); // CHECK: store volatile i32 9
}

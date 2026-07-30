/// Test determinisim when serializing anonymous decls. Create enough Decls in
/// DeclContext that can overflow the small storage of SmallPtrSet to make sure
/// the serialization does not rely on iteration order of SmallPtrSet.
// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.dir/cache -triple x86_64-apple-macosx10.11.0 \
// RUN:   -I%t.dir/headers %t.dir/main.m -fdisable-module-hash -Wno-visibility
// RUN: mv %t.dir/cache/A.pcm %t1.pcm
/// Check the order of the decls first. If LLVM_ENABLE_REVERSE_ITERATION is on,
/// it will fail the test early if the output is depending on the order of items
/// in containers that has non-deterministic orders.
// RUN: llvm-bcanalyzer --dump --disable-histogram %t1.pcm | FileCheck %s
// RUN: rm -rf %t.dir/cache
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.dir/cache -triple x86_64-apple-macosx10.11.0 \
// RUN:   -I%t.dir/headers %t.dir/main.m -fdisable-module-hash -Wno-visibility
// RUN: mv %t.dir/cache/A.pcm %t2.pcm
// RUN: diff %t1.pcm %t2.pcm

/// Spot check entries to make sure they are in current ordering.
/// op6 encodes the anonymous decl number which should be in order.

/// NOTE: This test case is on determinism of TypeID for function declaration.
/// Change related to TypeID (or PredefinedTypeIDs) will affect the result and
/// will require update for this test case. Currently, TypeID is at op6 and the
/// test checks the IDs are in strict ordering.

// CHECK: <TYPE_FUNCTION_PROTO
// CHECK-NEXT: <DECL_PARM_VAR
// CHECK-SAME: op5=13
// CHECK-NEXT: <DECL_PARM_VAR
// CHECK-SAME: op5=14
// CHECK-NEXT: <DECL_PARM_VAR
// CHECK-SAME: op5=15
// CHECK-NEXT: <DECL_PARM_VAR
// CHECK-SAME: op5=16

/// Decl records start at 43
// CHECK: <DECL_RECORD
// CHECK-SAME: op5=54
// CHECK-NEXT: <DECL_RECORD
// CHECK-SAME: op5=55
// CHECK-NEXT: <DECL_RECORD
// CHECK-SAME: op5=56
// CHECK-NEXT: <DECL_RECORD
// CHECK-SAME: op5=57

//--- headers/a.h
void f(struct A0 *a0,
       struct A1 *a1,
       struct A2 *a2,
       struct A3 *a3,
       struct A4 *a4,
       struct A5 *a5,
       struct A6 *a6,
       struct A7 *a7,
       struct A8 *a8,
       struct A9 *a9,
       struct A10 *a10,
       struct A11 *a11,
       struct A12 *a12,
       struct A13 *a13,
       struct A14 *a14,
       struct A15 *a15,
       struct A16 *a16,
       struct A17 *a17,
       struct A18 *a18,
       struct A19 *a19,
       struct A20 *a20,
       struct A21 *a21,
       struct A22 *a22,
       struct A23 *a23,
       struct A24 *a24,
       struct A25 *a25,
       struct A26 *a26,
       struct A27 *a27,
       struct A28 *a28,
       struct A29 *a29,
       struct A30 *a30,
       struct A31 *a31,
       struct A32 *a32,
       struct A33 *a33,
       struct A34 *a34,
       struct A35 *a35,
       struct A36 *a36,
       struct A37 *a37,
       struct A38 *a38,
       struct A39 *a39,
       struct A40 *a40);


//--- headers/module.modulemap

module A {
  header "a.h"
}

//--- main.m

#import <a.h>


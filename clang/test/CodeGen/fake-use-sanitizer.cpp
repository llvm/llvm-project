// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness -fsanitize=null -fsanitize-trap=null -o - | FileCheck --check-prefixes=CHECK,NULL --implicit-check-not=ubsantrap %s
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness -o - | FileCheck %s

// With -fextend-variable-liveness, the compiler previously generated a fake.use of any
// reference variable at the end of the scope in which its alloca exists. This
// caused two issues, where we would get fake uses for uninitialized variables
// if that variable was declared after an early-return, and UBSan's null checks
// would complain about this.
// This test verifies that UBSan does not produce null-checks for arguments to
// llvm.fake.use, and that fake uses are not emitted for a variable on paths
// it has not been declared.

struct A { short s1, s2; };
extern long& getA();

void foo()
{
  auto& va = getA();
  if (va < 5)
    return;

  auto& vb = getA();
}

// CHECK-LABEL:  define{{.*}}foo
// CHECK:  [[VA_CALL:%.+]] = call{{.*}} ptr @_Z4getAv()

/// We check here for the first UBSan check for "va".
// NULL:   [[VA_ISNULL:%.+]] = icmp ne ptr [[VA_CALL]], null
// NULL:   br i1 [[VA_ISNULL]], label %{{[^,]+}}, label %[[VA_TRAP:[^,]+]]
// NULL: [[VA_TRAP]]:
// NULL:   call void @llvm.ubsantrap(

// CHECK:       [[VA_PTR:%.+]] = load ptr, ptr %va
// CHECK-NEXT:  [[VA_CMP:%.+]] = load i64, ptr [[VA_PTR]]
// CHECK-NEXT:  [[VA_CMP_RES:%.+]] = icmp slt i64 [[VA_CMP]], 5
// CHECK-NEXT:  br i1 [[VA_CMP_RES]], label %[[EARLY_EXIT:[^,]+]], label %[[NOT_EARLY_EXIT:[^,]+]]

// CHECK: [[EARLY_EXIT]]:
// CHECK:   br label %cleanup

/// The fake use for "vb" only appears on the path where its declaration is
/// reached.
// CHECK:     [[NOT_EARLY_EXIT]]:
// CHECK:  [[VB_CALL:%.+]] = call{{.*}} ptr @_Z4getAv()

/// We check here for the second UBSan check for "vb".
// NULL:   [[VB_ISNULL:%.+]] = icmp ne ptr [[VB_CALL]], null
// NULL:   br i1 [[VB_ISNULL]], label %{{[^,]+}}, label %[[VB_TRAP:[^,]+]]
// NULL: [[VB_TRAP]]:
// NULL:   call void @llvm.ubsantrap(

// CHECK:       [[VB_FAKE_USE:%.+]] = load ptr, ptr %vb
// CHECK-NEXT:  call void (...) @llvm.fake.use(ptr [[VB_FAKE_USE]])
// CHECK:       br label %cleanup

// CHECK:     cleanup:
// CHECK:       [[VA_FAKE_USE:%.+]] = load ptr, ptr %va
// CHECK-NEXT:  call void (...) @llvm.fake.use(ptr [[VA_FAKE_USE]])

// NULL: declare void @llvm.ubsantrap

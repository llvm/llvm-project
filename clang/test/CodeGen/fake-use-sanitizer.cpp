// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness -fsanitize=null -fsanitize-trap=null -o - | FileCheck --check-prefixes=CHECK,NULL --implicit-check-not=ubsantrap %s
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness -fsanitize=memory -o - | FileCheck %s --check-prefixes=MEMORY
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


// MSan regression test from https://github.com/llvm/llvm-project/issues/225425
// This test (not the tests above) was generated with update_cc_test_checks.py.
//
// A fake_use will be inserted for x. MSan shouldn't check its parameter.
//
// MEMORY-LABEL: define dso_local noundef i32 @main(
// MEMORY-SAME: ) #[[ATTR0:[0-9]+]] {
// MEMORY-NEXT:  [[ENTRY:.*:]]
// MEMORY-NEXT:    call void @llvm.donothing()
// MEMORY-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
// MEMORY-NEXT:    [[TMP0:%.*]] = ptrtoint ptr [[RETVAL]] to i64
// MEMORY-NEXT:    [[TMP1:%.*]] = xor i64 [[TMP0]], 87960930222080
// MEMORY-NEXT:    [[TMP2:%.*]] = inttoptr i64 [[TMP1]] to ptr
// MEMORY-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[TMP2]], i8 -1, i64 4, i1 false)
// MEMORY-NEXT:    [[X:%.*]] = alloca i8, align 1
// MEMORY-NEXT:    [[TMP3:%.*]] = ptrtoint ptr [[RETVAL]] to i64
// MEMORY-NEXT:    [[TMP4:%.*]] = xor i64 [[TMP3]], 87960930222080
// MEMORY-NEXT:    [[TMP5:%.*]] = inttoptr i64 [[TMP4]] to ptr
// MEMORY-NEXT:    store i32 0, ptr [[TMP5]], align 4
// MEMORY-NEXT:    store i32 0, ptr [[RETVAL]], align 4
// MEMORY-NEXT:    call void @llvm.lifetime.start.p0(ptr [[X]]) #[[ATTR6:[0-9]+]]
// MEMORY-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[X]] to i64
// MEMORY-NEXT:    [[TMP7:%.*]] = xor i64 [[TMP6]], 87960930222080
// MEMORY-NEXT:    [[TMP8:%.*]] = inttoptr i64 [[TMP7]] to ptr
// MEMORY-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[TMP8]], i8 -1, i64 1, i1 false)
// MEMORY-NEXT:    [[FAKE_USE:%.*]] = load i8, ptr [[X]], align 1
// MEMORY-NEXT:    [[TMP9:%.*]] = ptrtoint ptr [[X]] to i64
// MEMORY-NEXT:    [[TMP10:%.*]] = xor i64 [[TMP9]], 87960930222080
// MEMORY-NEXT:    [[TMP11:%.*]] = inttoptr i64 [[TMP10]] to ptr
// MEMORY-NEXT:    [[_MSLD:%.*]] = load i8, ptr [[TMP11]], align 1
// MEMORY-NEXT:    notail call void (...) @llvm.fake.use(i8 [[FAKE_USE]]) #[[ATTR6]]
// MEMORY-NEXT:    call void @llvm.lifetime.end.p0(ptr [[X]]) #[[ATTR6]]
// MEMORY-NEXT:    ret i32 0
int main() {
  unsigned char x;
  return 0;
}

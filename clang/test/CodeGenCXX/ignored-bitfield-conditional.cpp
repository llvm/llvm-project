// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct S {
  int field1 : 5;
  int field2 : 6;
  int field3 : 3;
};

void use(bool cond, struct S s1, struct S s2, int val1, int val2) {
  // CHECK-LABEL: define {{.*}}use{{.*}}(
  // CHECK: %[[S1:.+]] = alloca %struct.S
  // CHECK: %[[S2:.+]] = alloca %struct.S
  // CHECK: %[[COND:.+]] = alloca i8
  // CHECK: %[[VAL1:.+]] = alloca i32
  // CHECK: %[[VAL2:.+]] = alloca i32

  cond ? s1.field1 = val1 : s1.field2 = val2;
  // Condition setup, branch.
  // CHECK: %[[CONDLD:.+]] = load i8, ptr %[[COND]]
  // CHECK: %[[TO_BOOL:.+]] = trunc i8 %[[CONDLD]] to i1
  // CHECK: br i1 %[[TO_BOOL]], label %[[TRUE:.+]], label %[[FALSE:.+]]

  // 'True', branch set the BF, branch to 'end'.
  // CHECK: [[TRUE]]:
  // CHECK: %[[VAL1LD:.+]] = load i32, ptr %[[VAL1]]
  // CHECK: %[[VAL1TRUNC:.+]] = trunc i32 %[[VAL1LD]] to i16
  // CHECK: %[[BF_LOAD:.+]] = load i16, ptr %[[S1]]
  // CHECK: %[[BF_VAL:.+]] = and i16 %[[VAL1TRUNC]], 31
  // CHECK: %[[BF_CLEAR:.+]] = and i16 %[[BF_LOAD]], -32
  // CHECK: %[[BF_SET:.+]] = or i16 %[[BF_CLEAR]], %[[BF_VAL]]
  // CHECK: store i16 %[[BF_SET]], ptr %[[S1]]
  // CHECK: br label %[[END:.+]]

  // 'False', branch set the OTHER BF, branch to 'end'.
  // CHECK: [[FALSE]]:
  // CHECK: %[[VAL2LD:.+]] = load i32, ptr %[[VAL2]]
  // CHECK: %[[VAL2TRUNC:.+]] = trunc i32 %[[VAL2LD]] to i16
  // CHECK: %[[BF_LOAD:.+]] = load i16, ptr %[[S1]]
  // CHECK: %[[BF_VAL:.+]] = and i16 %[[VAL2TRUNC]], 63 
  // CHECK: %[[BF_SHIFT:.+]] = shl i16 %[[BF_VAL]], 5
  // CHECK: %[[BF_CLEAR:.+]] = and i16 %[[BF_LOAD]], -2017
  // CHECK: %[[BF_SET:.+]] = or i16 %[[BF_CLEAR]], %[[BF_SHIFT]]
  // CHECK: store i16 %[[BF_SET]], ptr %[[S1]]
  // CHECK: br label %[[END:.+]]

  // CHECK: [[END]]:
  // There is nothing in the 'end' block associated with this, but it is the
  // 'continuation' block for the rest of the function.

  // Same test, has a no-op cast and parens.
  (void)(cond ? s2.field1 = val1 : s2.field2 = val2);
  // Condition setup, branch.
  // CHECK: %[[CONDLD:.+]] = load i8, ptr %[[COND]]
  // CHECK: %[[TO_BOOL:.+]] = trunc i8 %[[CONDLD]] to i1
  // CHECK: br i1 %[[TO_BOOL]], label %[[TRUE:.+]], label %[[FALSE:.+]]

  // 'True', branch set the BF, branch to 'end'.
  // CHECK: [[TRUE]]:
  // CHECK: %[[VAL1LD:.+]] = load i32, ptr %[[VAL1]]
  // CHECK: %[[VAL1TRUNC:.+]] = trunc i32 %[[VAL1LD]] to i16
  // CHECK: %[[BF_LOAD:.+]] = load i16, ptr %[[S2]]
  // CHECK: %[[BF_VAL:.+]] = and i16 %[[VAL1TRUNC]], 31
  // CHECK: %[[BF_CLEAR:.+]] = and i16 %[[BF_LOAD]], -32
  // CHECK: %[[BF_SET:.+]] = or i16 %[[BF_CLEAR]], %[[BF_VAL]]
  // CHECK: store i16 %[[BF_SET]], ptr %[[S2]]
  // CHECK: br label %[[END:.+]]

  // 'False', branch set the OTHER BF, branch to 'end'.
  // CHECK: [[FALSE]]:
  // CHECK: %[[VAL2LD:.+]] = load i32, ptr %[[VAL2]]
  // CHECK: %[[VAL2TRUNC:.+]] = trunc i32 %[[VAL2LD]] to i16
  // CHECK: %[[BF_LOAD:.+]] = load i16, ptr %[[S2]]
  // CHECK: %[[BF_VAL:.+]] = and i16 %[[VAL2TRUNC]], 63 
  // CHECK: %[[BF_SHIFT:.+]] = shl i16 %[[BF_VAL]], 5
  // CHECK: %[[BF_CLEAR:.+]] = and i16 %[[BF_LOAD]], -2017
  // CHECK: %[[BF_SET:.+]] = or i16 %[[BF_CLEAR]], %[[BF_SHIFT]]
  // CHECK: store i16 %[[BF_SET]], ptr %[[S2]]
  // CHECK: br label %[[END:.+]]

  // CHECK: [[END]]:
  // CHECK-NOT: phi
  // There is nothing in the 'end' block associated with this, but it is the
  // 'continuation' block for the rest of the function.

}


void use2(bool cond1, bool cond2, struct S s1, int val1, int val2, int val3) {
  // CHECK-LABEL: define {{.*}}use2{{.*}}(
  // CHECK: %[[S1:.+]] = alloca %struct.S
  // CHECK: %[[COND1:.+]] = alloca i8
  // CHECK: %[[COND2:.+]] = alloca i8
  // CHECK: %[[VAL1:.+]] = alloca i32
  // CHECK: %[[VAL2:.+]] = alloca i32
  // CHECK: %[[VAL3:.+]] = alloca i32

  cond1 ? s1.field1 = val1 : cond2 ? s1.field2 = val2 : s1.field3 = val3;
  // First Condition setup, branch.
  // CHECK: %[[CONDLD:.+]] = load i8, ptr %[[COND1]]
  // CHECK: %[[TO_BOOL:.+]] = trunc i8 %[[CONDLD]] to i1
  // CHECK: br i1 %[[TO_BOOL]], label %[[TRUE:.+]], label %[[FALSE:.+]]

  // First 'True' branch, sets field1 to val1.
  // CHECK: [[TRUE]]:
  // CHECK: %[[VAL1LD:.+]] = load i32, ptr %[[VAL1]]
  // CHECK: %[[VAL1TRUNC:.+]] = trunc i32 %[[VAL1LD]] to i16
  // CHECK: %[[BF_LOAD:.+]] = load i16, ptr %[[S1]]
  // CHECK: %[[BF_VAL:.+]] = and i16 %[[VAL1TRUNC]], 31
  // CHECK: %[[BF_CLEAR:.+]] = and i16 %[[BF_LOAD]], -32
  // CHECK: %[[BF_SET:.+]] = or i16 %[[BF_CLEAR]], %[[BF_VAL]]
  // CHECK: store i16 %[[BF_SET]], ptr %[[S1]]
  // CHECK: br label %[[END:.+]]

  // First 'False' branch, starts second ignored expression.
  // CHECK: [[FALSE]]:
  // CHECK: %[[CONDLD:.+]] = load i8, ptr %[[COND2]]
  // CHECK: %[[TO_BOOL:.+]] = trunc i8 %[[CONDLD]] to i1
  // CHECK: br i1 %[[TO_BOOL]], label %[[TRUE2:.+]], label %[[FALSE2:.+]]

  // Second 'True' branch, sets field2 to val2.
  // CHECK: [[TRUE2]]:
  // CHECK: %[[VAL2LD:.+]] = load i32, ptr %[[VAL2]]
  // CHECK: %[[VAL2TRUNC:.+]] = trunc i32 %[[VAL2LD]] to i16
  // CHECK: %[[BF_LOAD:.+]] = load i16, ptr %[[S1]]
  // CHECK: %[[BF_VAL:.+]] = and i16 %[[VAL2TRUNC]], 63 
  // CHECK: %[[BF_SHIFT:.+]] = shl i16 %[[BF_VAL]], 5
  // CHECK: %[[BF_CLEAR:.+]] = and i16 %[[BF_LOAD]], -2017
  // CHECK: %[[BF_SET:.+]] = or i16 %[[BF_CLEAR]], %[[BF_SHIFT]]
  // CHECK: store i16 %[[BF_SET]], ptr %[[S1]]
  // CHECK: br label %[[END:.+]]

  // Second 'False' branch, sets field3 to val3.
  // CHECK: [[FALSE2]]:
  // CHECK: %[[VAL3LD:.+]] = load i32, ptr %[[VAL3]]
  // CHECK: %[[VAL3TRUNC:.+]] = trunc i32 %[[VAL3LD]] to i16
  // CHECK: %[[BF_LOAD:.+]] = load i16, ptr %[[S1]]
  // CHECK: %[[BF_VAL:.+]] = and i16 %[[VAL3TRUNC]], 7
  // CHECK: %[[BF_SHIFT:.+]] = shl i16 %[[BF_VAL]], 11
  // CHECK: %[[BF_CLEAR:.+]] = and i16 %[[BF_LOAD]], -14337
  // CHECK: %[[BF_SET:.+]] = or i16 %[[BF_CLEAR]], %[[BF_SHIFT]]
  // CHECK: store i16 %[[BF_SET]], ptr %[[S1]]
  // CHECK: br label %[[END:.+]]

  // CHECK: [[END]]:
  // CHECK-NOT: phi
  // Nothing left to do here.
}

// Catch the end of the IR. This prevents the CHECK-NOT above from matching a
// spurious "phi" in file paths printed in the output.
// CHECK-LABEL: attributes

// RUN: %clang_cc1 %s -triple s390x-ibm-zos -emit-llvm -o - | FileCheck %s
// REQUIRES: systemz-registered-target

// The aliasee of an alias attribute is a symbol name, so it must use the mapped
// name when the aliasee is mapped. A mapped alias is emitted under its mapped
// name.

// Mapped aliasee; the alias itself is not mapped.
#pragma map(t1, "T1")
void t1(void) {}
void a1(void) __attribute__((alias("T1")));

// Mapped alias, pragma before the alias declaration.
void t2(void) {}
#pragma map(a2, "A2")
void a2(void) __attribute__((alias("t2")));

// Mapped alias and mapped aliasee.
#pragma map(t4, "T4")
void t4(void) {}
#pragma map(a4, "A4")
void a4(void) __attribute__((alias("T4")));

// Variables: mapped aliasee and mapped alias, with the pragma before the alias
// declaration.
#pragma map(v5, "V5")
int v5 = 5;
#pragma map(va5, "VA5")
extern int va5 __attribute__((alias("V5")));

// Variable aliasee mapped; the alias itself is not mapped.
#pragma map(v7, "V7")
int v7 = 7;
extern int va7 __attribute__((alias("V7")));

int use(void) {
  a1();
  a2();
  a4();
  return va5 + va7;
}

// CHECK-DAG: @VA5 = {{.*}}alias i32, ptr @V5
// CHECK-DAG: @va7 = {{.*}}alias i32, ptr @V7
// CHECK-DAG: @a1 = {{.*}}alias void (), ptr @T1
// CHECK-DAG: @A2 = {{.*}}alias void (), ptr @t2
// CHECK-DAG: @A4 = {{.*}}alias void (), ptr @T4
// CHECK-DAG: define{{.*}} void @T1()
// CHECK-DAG: define{{.*}} void @t2()
// CHECK-DAG: define{{.*}} void @T4()
// CHECK:     define{{.*}} i32 @use()
// CHECK:       call void @a1()
// CHECK-NEXT:  call void @A2()
// CHECK-NEXT:  call void @A4()
// CHECK:       load i32, ptr @VA5
// CHECK:       load i32, ptr @va7

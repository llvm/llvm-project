// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple %ms_abi_triple -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=MSABI

// Should be 3 hello strings, two global (of different sizes), the rest are
// shared.

// ITANIUM: @.str = private unnamed_addr constant [6 x i8] c"hello\00"
// MSABI: @"??_C@_05CJBACGMB@hello?$AA@" = linkonce_odr dso_local unnamed_addr constant [6 x i8] c"hello\00", comdat
// ITANIUM: @f1.x = internal global ptr @.str
// MSABI: @f1.x = internal global ptr @"??_C@_05CJBACGMB@hello?$AA@"
// CHECK: @f2.x = internal global [6 x i8] c"hello\00"
// CHECK: @f3.x = internal global [8 x i8] c"hello\00\00\00"
// ITANIUM: @f4.x = internal global %struct.s { ptr @.str }
// MSABI: @f4.x = internal global %struct.s { ptr @"??_C@_05CJBACGMB@hello?$AA@" }
// CHECK: @x = {{(dso_local )?}}global [3 x i8] c"ola"

void bar(const char *);

// CHECK-LABEL: define {{.*}}void @f0()
void f0(void) {
  bar("hello");
  // ITANIUM: call {{.*}}void @bar({{.*}} @.str
  // MSABI: call {{.*}}void @bar({{.*}} @"??_C@_05CJBACGMB@hello?$AA@"
}

// CHECK-LABEL: define {{.*}}void @f1()
void f1(void) {
  static char *x = "hello";
  bar(x);
  // CHECK: [[T1:%.*]] = load ptr, ptr @f1.x
  // CHECK: call {{.*}}void @bar(ptr noundef [[T1:%.*]])
}

// CHECK-LABEL: define {{.*}}void @f2()
void f2(void) {
  static char x[] = "hello";
  bar(x);
  // CHECK: call {{.*}}void @bar({{.*}} @f2.x
}

// CHECK-LABEL: define {{.*}}void @f3()
void f3(void) {
  static char x[8] = "hello";
  bar(x);
  // CHECK: call {{.*}}void @bar({{.*}} @f3.x
}

void gaz(void *);

// CHECK-LABEL: define {{.*}}void @f4()
void f4(void) {
  static struct s {
    char *name;
  } x = { "hello" };
  gaz(&x);
  // CHECK: call {{.*}}void @gaz({{.*}} @f4.x
}

char x[3] = "ola";

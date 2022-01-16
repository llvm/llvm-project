// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple %ms_abi_triple -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=MSABI

// Should be 3 hello strings, two global (of different sizes), the rest are
// shared.

// CHECK: @align = {{(dso_local )?}}global i8 [[ALIGN:[0-9]+]]
// ITANIUM: @.str = private unnamed_addr constant [6 x i8] c"hello\00"
// MSABI: @"??_C@_05CJBACGMB@hello?$AA@" = linkonce_odr dso_local unnamed_addr constant [6 x i8] c"hello\00", comdat, align 1
// ITANIUM: @f1.x = internal global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i32 0, i32 0)
// MSABI: @f1.x = internal global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @"??_C@_05CJBACGMB@hello?$AA@", i32 0, i32 0)
// CHECK: @f2.x = internal global [6 x i8] c"hello\00", align [[ALIGN]]
// CHECK: @f3.x = internal global [8 x i8] c"hello\00\00\00", align [[ALIGN]]
// ITANIUM: @f4.x = internal global %struct.s { i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i32 0, i32 0) }
// MSABI: @f4.x = internal global %struct.s { i8* getelementptr inbounds ([6 x i8], [6 x i8]* @"??_C@_05CJBACGMB@hello?$AA@", i32 0, i32 0) }
// CHECK: @x = {{(dso_local )?}}global [3 x i8] c"ola", align [[ALIGN]]

// XFAIL: hexagon
// Hexagon aligns arrays of size 8+ bytes to a 64-bit boundary, which
// fails the check for "@f3.x = ... align [ALIGN]", since ALIGN is derived
// from the alignment of a single i8, which is still 1.

#if defined(__s390x__)
unsigned char align = 2;
#else
unsigned char align = 1;
#endif

void bar(const char *);

// CHECK-LABEL: define {{.*}}void @f0()
void f0() {
  bar("hello");
  // ITANIUM: call {{.*}}void @bar({{.*}} @.str
  // MSABI: call {{.*}}void @bar({{.*}} @"??_C@_05CJBACGMB@hello?$AA@"
}

// CHECK-LABEL: define {{.*}}void @f1()
void f1() {
  static char *x = "hello";
  bar(x);
  // CHECK: [[T1:%.*]] = load i8*, i8** @f1.x
  // CHECK: call {{.*}}void @bar(i8* noundef [[T1:%.*]])
}

// CHECK-LABEL: define {{.*}}void @f2()
void f2() {
  static char x[] = "hello";
  bar(x);
  // CHECK: call {{.*}}void @bar({{.*}} @f2.x
}

// CHECK-LABEL: define {{.*}}void @f3()
void f3() {
  static char x[8] = "hello";
  bar(x);
  // CHECK: call {{.*}}void @bar({{.*}} @f3.x
}

void gaz(void *);

// CHECK-LABEL: define {{.*}}void @f4()
void f4() {
  static struct s {
    char *name;
  } x = { "hello" };
  gaz(&x);
  // CHECK: call {{.*}}void @gaz({{.*}} @f4.x
}

char x[3] = "ola";


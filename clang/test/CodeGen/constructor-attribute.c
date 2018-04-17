// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=WITHOUTATEXIT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fregister-global-dtors-with-atexit -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=CXAATEXIT --check-prefix=WITHATEXIT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fno-use-cxa-atexit -fregister-global-dtors-with-atexit -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ATEXIT --check-prefix=WITHATEXIT %s

// WITHOUTATEXIT: global_ctors{{.*}}@A{{.*}}@C
// WITHOUTATEXIT: @llvm.global_dtors = appending global [5 x { i32, void ()*, i8* }]{{.*}}@B{{.*}}@E{{.*}}@F{{.*}}@G{{.*}}@D
// WITHATEXIT: @llvm.global_ctors = appending global [5 x { i32, void ()*, i8* }]{{.*}}i32 65535, void ()* @A,{{.*}}i32 65535, void ()* @C,{{.*}}i32 123, void ()* @__GLOBAL_init_123,{{.*}}i32 789, void ()* @[[GLOBAL_INIT_789:__GLOBAL_init_789.[0-9]+]],{{.*}}i32 65535, void ()* @__GLOBAL_init_65535,
// WITHATEXIT-NOT: global_dtors

// CHECK: define void @A()
// CHECK: define void @B()
// CHECK: define internal void @E()
// CHECK: define internal void @F()
// CHECK: define internal void @G()
// CHECK: define i32 @__GLOBAL_init_789(i32 %{{.*}})
// CHECK: define internal void @C()
// CHECK: define internal void @D()
// CHECK: define i32 @main()
// CHECK: define internal i32 @foo()
// WITHOUTATEXIT-NOT: define

// WITHATEXIT: define internal void @__GLOBAL_init_123(){{.*}}section "__TEXT,__StaticInit,regular,pure_instructions"
// CXAATEXIT: call i32 @__cxa_atexit(void (i8*)* bitcast (void ()* @E to void (i8*)*), i8* null, i8* @__dso_handle)
// CXAATEXIT: call i32 @__cxa_atexit(void (i8*)* bitcast (void ()* @G to void (i8*)*), i8* null, i8* @__dso_handle)
// ATEXIT: call i32 @atexit(void ()* @E)
// ATEXIT: call i32 @atexit(void ()* @G)

// WITHATEXIT: define internal void @[[GLOBAL_INIT_789]](){{.*}}section "__TEXT,__StaticInit,regular,pure_instructions"
// CXAATEXIT: call i32 @__cxa_atexit(void (i8*)* bitcast (void ()* @F to void (i8*)*), i8* null, i8* @__dso_handle)
// ATEXIT: call i32 @atexit(void ()* @F)

// WITHATEXIT: define internal void @__GLOBAL_init_65535(){{.*}}section "__TEXT,__StaticInit,regular,pure_instructions"
// CXAATEXIT: call i32 @__cxa_atexit(void (i8*)* bitcast (void ()* @B to void (i8*)*), i8* null, i8* @__dso_handle)
// CXAATEXIT: call i32 @__cxa_atexit(void (i8*)* bitcast (void ()* @D to void (i8*)*), i8* null, i8* @__dso_handle)
// ATEXIT: call i32 @atexit(void ()* @B)
// ATEXIT: call i32 @atexit(void ()* @D)

int printf(const char *, ...);

void A() __attribute__((constructor));
void B() __attribute__((destructor));

void A() {
  printf("A\n");
}

void B() {
  printf("B\n");
}

static void C() __attribute__((constructor));

static void D() __attribute__((destructor));

static __attribute__((destructor(123))) void E() {
}

static __attribute__((destructor(789))) void F() {
}

static __attribute__((destructor(123))) void G() {
}

// Test that this function doesn't collide with the synthesized constructor
// function for destructors with priority 789.
int __GLOBAL_init_789(int a) {
  return a * a;
}

static int foo() {
  return 10;
}

static void C() {
  printf("A: %d\n", foo());
}

static void D() {
  printf("B\n");
}

int main() {
  return 0;
}

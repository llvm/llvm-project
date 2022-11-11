// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -x c++ -o - %s | FileCheck %s
#if !__has_feature(kcfi)
#error Missing kcfi?
#endif

/// Must emit __kcfi_typeid symbols for address-taken function declarations
// CHECK: module asm ".weak __kcfi_typeid_[[F4:[a-zA-Z0-9_]+]]"
// CHECK: module asm ".set __kcfi_typeid_[[F4]], [[#%d,HASH:]]"
/// Must not __kcfi_typeid symbols for non-address-taken declarations
// CHECK-NOT: module asm ".weak __kcfi_typeid_{{f6|_Z2f6v}}"
typedef int (*fn_t)(void);

// CHECK: define dso_local{{.*}} i32 @{{f1|_Z2f1v}}(){{.*}} !kcfi_type ![[#TYPE:]]
int f1(void) { return 0; }

// CHECK: define dso_local{{.*}} i32 @{{f2|_Z2f2v}}(){{.*}} !kcfi_type ![[#TYPE2:]]
unsigned int f2(void) { return 2; }

// CHECK-LABEL: define dso_local{{.*}} i32 @{{__call|_Z6__callPFivE}}(ptr{{.*}} %f)
int __call(fn_t f) __attribute__((__no_sanitize__("kcfi"))) {
  // CHECK-NOT: call{{.*}} i32 %{{.}}(){{.*}} [ "kcfi"
  return f();
}

// CHECK: define dso_local{{.*}} i32 @{{call|_Z4callPFivE}}(ptr{{.*}} %f){{.*}}
int call(fn_t f) {
  // CHECK: call{{.*}} i32 %{{.}}(){{.*}} [ "kcfi"(i32 [[#HASH]]) ]
  return f();
}

// CHECK-DAG: define internal{{.*}} i32 @{{f3|_ZL2f3v}}(){{.*}} !kcfi_type ![[#TYPE]]
static int f3(void) { return 1; }

// CHECK-DAG: declare !kcfi_type ![[#TYPE]]{{.*}} i32 @[[F4]]()
extern int f4(void);

/// Must not emit !kcfi_type for non-address-taken local functions
// CHECK: define internal{{.*}} i32 @{{f5|_ZL2f5v}}()
// CHECK-NOT: !kcfi_type
// CHECK-SAME: {
static int f5(void) { return 2; }

// CHECK-DAG: declare !kcfi_type ![[#TYPE]]{{.*}} i32 @{{f6|_Z2f6v}}()
extern int f6(void);

int test(void) {
  return call(f1) +
         __call((fn_t)f2) +
         call(f3) +
         call(f4) +
         f5() +
         f6();
}

// CHECK-DAG: ![[#]] = !{i32 4, !"kcfi", i32 1}
// CHECK-DAG: ![[#TYPE]] = !{i32 [[#HASH]]}
// CHECK-DAG: ![[#TYPE2]] = !{i32 [[#%d,HASH2:]]}

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -DORIG_ATTR_SYN -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fpatchable-function-entry-offset=3 -DORIG_ATTR_SYN -o - %s | FileCheck %s --check-prefixes=CHECK,OFFSET
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=kcfi -fpatchable-function-entry-offset=3 -o - %s | FileCheck %s --check-prefixes=CHECK,OFFSET

// Note that the interleving of functions, which normally would be in sequence,
// is due to the fact that Clang outputs them in a non-sequential order.

#if !__has_feature(kcfi)
#error Missing kcfi?
#endif

#ifdef ORIG_ATTR_SYN
#define __cfi_salt       __attribute__((cfi_salt("pepper")))
#define __cfi_salt_empty __attribute__((cfi_salt("")))
#else
#define __cfi_salt       [[clang::cfi_salt("pepper")]]
#define __cfi_salt_empty [[clang::cfi_salt("")]]
#endif

typedef int (*fn_t)(void);
typedef int (* __cfi_salt fn_salt_t)(void);
typedef int (* __cfi_salt_empty fn_salt_empty_t)(void);

typedef unsigned int (*ufn_t)(void);
typedef unsigned int (* __cfi_salt ufn_salt_t)(void);

/// Must emit __kcfi_typeid symbols for address-taken function declarations
// CHECK: module asm ".weak __kcfi_typeid_[[F4:[a-zA-Z0-9_]+]]"
// CHECK: module asm ".set __kcfi_typeid_[[F4]], [[#%d,LOW_SODIUM_HASH:]]"
// CHECK: module asm ".weak __kcfi_typeid_[[F4_SALT:[a-zA-Z0-9_]+]]"
// CHECK: module asm ".set __kcfi_typeid_[[F4_SALT]], [[#%d,ASM_SALTY_HASH:]]"

/// Must not __kcfi_typeid symbols for non-address-taken declarations
// CHECK-NOT: module asm ".weak __kcfi_typeid_f6"

int f1(void);
int f1_salt(void) __cfi_salt;

unsigned int f2(void);
unsigned int f2_salt(void) __cfi_salt;

static int f3(void);
static int f3_salt(void) __cfi_salt;

extern int f4(void);
extern int f4_salt(void) __cfi_salt;

static int f5(void);
static int f5_salt(void) __cfi_salt;

extern int f6(void);
extern int f6_salt(void) __cfi_salt;

int f8(void);
int f8_salt_empty(void) __cfi_salt_empty;

struct cfi_struct {
  fn_t __cfi_salt fptr;
  fn_salt_t td_fptr;
  fn_salt_empty_t td_empty_fptr;
};

int f7_salt(struct cfi_struct *ptr);
int f7_typedef_salt(struct cfi_struct *ptr);

// CHECK-LABEL: @__call
// CHECK:         call{{.*}} i32
// CHECK-NOT:     "kcfi"
// CHECK-SAME:    ()
__attribute__((__no_sanitize__("kcfi")))
int __call(fn_t f) {
  return f();
}

// CHECK-LABEL: @call
// CHECK:         call{{.*}} i32 %{{.}}(){{.*}} [ "kcfi"(i32 [[#LOW_SODIUM_HASH]]) ]
// CHECK-LABEL: @call_salt
// CHECK:         call{{.*}} i32 %{{.}}(){{.*}} [ "kcfi"(i32 [[#%d,SALTY_HASH:]]) ]
// CHECK-LABEL: @call_salt_ty
// CHECK:         call{{.*}} i32 %{{.}}(){{.*}} [ "kcfi"(i32 [[#SALTY_HASH]]) ]
int call(fn_t f) { return f(); }
int call_salt(fn_t __cfi_salt f) { return f(); }
int call_salt_ty(fn_salt_t f) { return f(); }
int call_salt_empty_ty(fn_salt_empty_t f) { return f(); }

// CHECK-LABEL: @ucall
// CHECK:         call{{.*}} i32 %{{.}}(){{.*}} [ "kcfi"(i32 [[#%d,LOW_SODIUM_UHASH:]]) ]
// CHECK-LABEL: @ucall_salt
// CHECK:         call{{.*}} i32 %{{.}}(){{.*}} [ "kcfi"(i32 [[#%d,SALTY_UHASH:]]) ]
// CHECK-LABEL: @ucall_salt_ty
// CHECK:         call{{.*}} i32 %{{.}}(){{.*}} [ "kcfi"(i32 [[#SALTY_UHASH]]) ]
unsigned int ucall(ufn_t f) { return f(); }
unsigned int ucall_salt(ufn_t __cfi_salt f) { return f(); }
unsigned int ucall_salt_ty(ufn_salt_t f) { return f(); }

int test1(struct cfi_struct *ptr) {
  return call(f1) +
         call_salt(f1_salt) +
         call_salt_ty(f1_salt) +

         __call((fn_t)f2) +
         __call((fn_t)f2_salt) +

         ucall(f2) +
         ucall_salt(f2_salt) +
         ucall_salt_ty(f2_salt) +

         call(f3) +
         call_salt(f3_salt) +
         call_salt_ty(f3_salt) +

         call(f4) +
         call_salt(f4_salt) +
         call_salt_ty(f4_salt) +

         f5() +
         f5_salt() +

         f6() +
         f6_salt() +

         f7_salt(ptr) +
         f7_typedef_salt(ptr) +

         f8() +
         f8_salt_empty();
}

// CHECK-LABEL: define dso_local{{.*}} i32 @f1(){{.*}} !kcfi_type
// CHECK-SAME:  ![[#LOW_SODIUM_TYPE:]]
// CHECK-LABEL: define dso_local{{.*}} i32 @f1_salt(){{.*}} !kcfi_type
// CHECK-SAME:  ![[#SALTY_TYPE:]]
int f1(void) { return 0; }
int f1_salt(void) __cfi_salt { return 0; }

// CHECK-LABEL: define dso_local{{.*}} i32 @f2(){{.*}} !kcfi_type
// CHECK-SAME:  ![[#LOW_SODIUM_UTYPE:]]
// CHECK: define dso_local{{.*}} i32 @f2_salt(){{.*}} !kcfi_type
// CHECK-SAME:  ![[#SALTY_UTYPE:]]
unsigned int f2(void) { return 2; }
unsigned int f2_salt(void) __cfi_salt { return 2; }

// CHECK-LABEL: define internal{{.*}} i32 @f3(){{.*}} !kcfi_type
// CHECK-SAME:  ![[#LOW_SODIUM_TYPE]]
// CHECK-LABEL: define internal{{.*}} i32 @f3_salt(){{.*}} !kcfi_type
// CHECK-SAME:  ![[#SALTY_TYPE]]
static int f3(void) { return 1; }
static int f3_salt(void) __cfi_salt { return 1; }

// CHECK: declare !kcfi_type ![[#LOW_SODIUM_TYPE]]{{.*}} i32 @[[F4]]()
// CHECK: declare !kcfi_type ![[#SALTY_TYPE]]{{.*}} i32 @[[F4_SALT]]()

/// Must not emit !kcfi_type for non-address-taken local functions
// CHECK-LABEL: define internal{{.*}} i32 @f5()
// CHECK-NOT:   !kcfi_type
// CHECK-SAME:  {
// CHECK-LABEL: define internal{{.*}} i32 @f5_salt()
// CHECK-NOT:   !kcfi_type
// CHECK-SAME:  {
static int f5(void) { return 2; }
static int f5_salt(void) __cfi_salt { return 2; }

// CHECK: declare !kcfi_type ![[#LOW_SODIUM_TYPE]]{{.*}} i32 @f6()
// CHECK: declare !kcfi_type ![[#SALTY_TYPE]]{{.*}} i32 @f6_salt()

// CHECK-LABEL: @f7_salt
// CHECK:         call{{.*}} i32 %{{.*}}() [ "kcfi"(i32 [[#SALTY_HASH]]) ]
// CHECK-LABEL: @f7_typedef_salt
// CHECK:         call{{.*}} i32 %{{.*}}() [ "kcfi"(i32 [[#SALTY_HASH]]) ]
int f7_salt(struct cfi_struct *ptr) { return ptr->fptr(); }
int f7_typedef_salt(struct cfi_struct *ptr) { return ptr->td_fptr(); }

// CHECK-LABEL: define dso_local{{.*}} i32 @f8(){{.*}} !kcfi_type
// CHECK-SAME:  ![[#LOW_SODIUM_TYPE:]]
// CHECK-LABEL: define dso_local{{.*}} i32 @f8_salt_empty(){{.*}} !kcfi_type
// CHECK-SAME:  ![[#LOW_SODIUM_TYPE:]]
int f8(void) { return 0; }
int f8_salt_empty(void) __cfi_salt_empty { return 0; }

// CHECK:  ![[#]] = !{i32 4, !"kcfi", i32 1}
// OFFSET: ![[#]] = !{i32 4, !"kcfi-offset", i32 3}
//
// CHECK:  ![[#LOW_SODIUM_TYPE]] = !{i32 [[#LOW_SODIUM_HASH]]}
// CHECK:  ![[#SALTY_TYPE]] = !{i32 [[#SALTY_HASH]]}
//
// CHECK:  ![[#LOW_SODIUM_UTYPE]] = !{i32 [[#LOW_SODIUM_UHASH]]}
// CHECK:  ![[#SALTY_UTYPE]] = !{i32 [[#SALTY_UHASH]]}

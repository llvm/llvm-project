// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -target-cpu core2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios9 -target-cpu cyclone -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7s-apple-ios9 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7k-apple-ios9 -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -x c++ -triple x86_64-apple-darwin10 -target-cpu core2 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CPPONLY
// RUN: %clang_cc1 -x c++ -triple arm64-apple-ios9 -target-cpu cyclone -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CPPONLY
// RUN: %clang_cc1 -x c++ -triple armv7-apple-darwin9 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CPPONLY
// RUN: %clang_cc1 -x c++ -triple armv7s-apple-ios9 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CPPONLY
// RUN: %clang_cc1 -x c++ -triple armv7k-apple-ios9 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CPPONLY

// Test tail call behavior when a swiftasynccall function is called
// from another swiftasynccall function.

#define SWIFTCALL __attribute__((swiftcall))
#define SWIFTASYNCCALL __attribute__((swiftasynccall))
#define ASYNC_CONTEXT __attribute__((swift_async_context))

// CHECK-LABEL: swifttailcc void {{.*}}async_leaf1{{.*}}(i8* swiftasync
SWIFTASYNCCALL void async_leaf1(char * ASYNC_CONTEXT ctx) {
  *ctx += 1;
}

// CHECK-LABEL: swifttailcc void {{.*}}async_leaf2{{.*}}(i8* swiftasync
SWIFTASYNCCALL void async_leaf2(char * ASYNC_CONTEXT ctx) {
  *ctx += 2;
}

#if __cplusplus
  #define MYBOOL bool
#else
  #define MYBOOL _Bool
#endif

// CHECK-LABEL: swifttailcc void {{.*}}async_branch{{.*}}i8* swiftasync
// CHECK: musttail call swifttailcc void @{{.*}}async_leaf1
// CHECK-NEXT: ret void
// CHECK: musttail call swifttailcc void @{{.*}}async_leaf2
// CHECK-NEXT: ret void
SWIFTASYNCCALL void async_branch(MYBOOL b, char * ASYNC_CONTEXT ctx) {
  if (b) {
    return async_leaf1(ctx);
  } else {
    return async_leaf2(ctx);
  }
}

// CHECK-LABEL: swifttailcc void {{.*}}async_not_all_tail
// CHECK-NOT:  musttail call swifttailcc void @{{.*}}async_leaf1
// CHECK:      call swifttailcc void @{{.*}}async_leaf1
// CHECK-NOT:  ret void
// CHECK:      musttail call swifttailcc void @{{.*}}async_leaf2
// CHECK-NEXT: ret void
SWIFTASYNCCALL void async_not_all_tail(char * ASYNC_CONTEXT ctx) {
  async_leaf1(ctx);
  return async_leaf2(ctx);
}

// CHECK-LABEL: swifttailcc void {{.*}}async_loop
// CHECK: musttail call swifttailcc void @{{.*}}async_leaf1
// CHECK-NEXT: ret void
// CHECK: musttail call swifttailcc void @{{.*}}async_leaf2
// CHECK-NEXT: ret void
// CHECK: musttail call swifttailcc void @{{.*}}async_loop
// CHECK-NEXT: ret void
SWIFTASYNCCALL void async_loop(unsigned u, char * ASYNC_CONTEXT ctx) {
  if (u == 0) {
    return async_leaf1(ctx);
  } else if (u == 1) {
    return async_leaf2(ctx);
  }
  return async_loop(u - 2, ctx);
}

// Forward-declaration + mutual recursion is okay.

SWIFTASYNCCALL void async_mutual_loop2(unsigned u, char * ASYNC_CONTEXT ctx);

// CHECK-LABEL: swifttailcc void {{.*}}async_mutual_loop1
// CHECK: musttail call swifttailcc void @{{.*}}async_leaf1
// CHECK-NEXT: ret void
// CHECK: musttail call swifttailcc void @{{.*}}async_leaf2
// CHECK-NEXT: ret void
// There is some bugginess around FileCheck's greediness/matching,
// so skipping the check for async_mutual_loop2 here.
SWIFTASYNCCALL void async_mutual_loop1(unsigned u, char * ASYNC_CONTEXT ctx) {
  if (u == 0) {
    return async_leaf1(ctx);
  } else if (u == 1) {
    return async_leaf2(ctx);
  }
  return async_mutual_loop2(u - 2, ctx);
}

// CHECK-LABEL: swifttailcc void {{.*}}async_mutual_loop2
// CHECK: musttail call swifttailcc void @{{.*}}async_leaf1
// CHECK-NEXT: ret void
// CHECK: musttail call swifttailcc void @{{.*}}async_leaf2
// CHECK-NEXT: ret void
// CHECK: musttail call swifttailcc void @{{.*}}async_mutual_loop1
// CHECK-NEXT: ret void
SWIFTASYNCCALL void async_mutual_loop2(unsigned u, char * ASYNC_CONTEXT ctx) {
  if (u == 0) {
    return async_leaf1(ctx);
  } else if (u == 1) {
    return async_leaf2(ctx);
  }
  return async_mutual_loop1(u - 2, ctx);
}

// When swiftasynccall functions are called by non-swiftasynccall functions,
// the call isn't marked as a tail call.

// CHECK-LABEL: swiftcc i8 {{.*}}sync_calling_async
// CHECK-NOT: tail call
// CHECK: call swifttailcc void @{{.*}}async_branch
// CHECK-NOT: tail call
// CHECK: call swifttailcc void @{{.*}}async_loop
SWIFTCALL char sync_calling_async(MYBOOL b, unsigned u) {
  char x = 'a';
  async_branch(b, &x);
  async_loop(u, &x);
  return x;
}

// CHECK-LABEL: i8 {{.*}}c_calling_async
// CHECK-NOT: tail call
// CHECK: call swifttailcc void @{{.*}}async_branch
// CHECK-NOT: tail call
// CHECK: call swifttailcc void @{{.*}}async_loop
char c_calling_async(MYBOOL b, unsigned u) {
  char x = 'a';
  async_branch(b, &x);
  async_loop(u, &x);
  return x;
}

#if __cplusplus
struct S {
  SWIFTASYNCCALL void (*fptr)(char * ASYNC_CONTEXT);

  SWIFTASYNCCALL void async_leaf_method(char * ASYNC_CONTEXT ctx) {
    *ctx += 1;
  }
  SWIFTASYNCCALL void async_nonleaf_method1(char * ASYNC_CONTEXT ctx) {
    return async_leaf_method(ctx);
  }
  SWIFTASYNCCALL void async_nonleaf_method2(char * ASYNC_CONTEXT ctx) {
    return this->async_leaf_method(ctx);
  }
};

SWIFTASYNCCALL void (S::*async_leaf_method_ptr)(char * ASYNC_CONTEXT) = &S::async_leaf_method;

// CPPONLY-LABEL: swifttailcc void {{.*}}async_struct_field_and_methods
// CPPONLY: musttail call swifttailcc void %{{[0-9]+}}
// CPPONLY: musttail call swifttailcc void @{{.*}}async_nonleaf_method1
// CPPONLY: musttail call swifttailcc void %{{[0-9]+}}
// CPPONLY: musttail call swifttailcc void @{{.*}}async_nonleaf_method2
// CPPONLY-NOT: musttail call swifttailcc void @{{.*}}async_leaf_method
// ^ TODO: Member pointers should also work.
SWIFTASYNCCALL void async_struct_field_and_methods(int i, S &sref, S *sptr) {
  char x = 'a';
  if (i == 0) {
    return (*sref.fptr)(&x);
  } else if (i == 1) {
    return sref.async_nonleaf_method1(&x);
  } else if (i == 2) {
    return (*(sptr->fptr))(&x);
  } else if (i == 3) {
    return sptr->async_nonleaf_method2(&x);
  } else if (i == 4) {
    return (sref.*async_leaf_method_ptr)(&x);
  }
  return (sptr->*async_leaf_method_ptr)(&x);
}

// CPPONLY-LABEL: define{{.*}} swifttailcc void @{{.*}}async_nonleaf_method1
// CPPONLY: musttail call swifttailcc void @{{.*}}async_leaf_method

// CPPONLY-LABEL: define{{.*}} swifttailcc void @{{.*}}async_nonleaf_method2
// CPPONLY: musttail call swifttailcc void @{{.*}}async_leaf_method
#endif

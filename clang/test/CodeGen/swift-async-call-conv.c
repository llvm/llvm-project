// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -target-cpu core2 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios9 -target-cpu cyclone -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7s-apple-ios9 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7k-apple-ios9 -emit-llvm -o - %s | FileCheck %s

// Test tail call behavior when a swiftasynccall function is called
// from another swiftasynccall function.

#define SWIFTCALL __attribute__((swiftcall))
#define SWIFTASYNCCALL __attribute__((swiftasynccall))
#define ASYNC_CONTEXT __attribute__((swift_async_context))

// CHECK-LABEL: swifttailcc void @async_leaf1(i8* swiftasync
SWIFTASYNCCALL void async_leaf1(char * ASYNC_CONTEXT ctx) {
  *ctx += 1;
}

// CHECK-LABEL: swifttailcc void @async_leaf2(i8* swiftasync
SWIFTASYNCCALL void async_leaf2(char * ASYNC_CONTEXT ctx) {
  *ctx += 2;
}

// CHECK-LABEL: swifttailcc void @async_branch
// CHECK: tail call swifttailcc void @async_leaf1
// CHECK: tail call swifttailcc void @async_leaf2
SWIFTASYNCCALL void async_branch(_Bool b, char * ASYNC_CONTEXT ctx) {
  if (b) {
    return async_leaf1(ctx);
  } else {
    return async_leaf2(ctx);
  }
}

// CHECK-LABEL: swifttailcc void @async_loop
// CHECK: tail call swifttailcc void @async_leaf1
// CHECK: tail call swifttailcc void @async_leaf2
// CHECK: tail call swifttailcc void @async_loop
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

// CHECK: swifttailcc void @async_mutual_loop
// CHECK: tail call swifttailcc void @async_leaf
// CHECK: tail call swifttailcc void @async_leaf
// CHECK: tail call swifttailcc void @async_mutual_loop
SWIFTASYNCCALL void async_mutual_loop1(unsigned u, char * ASYNC_CONTEXT ctx) {
  if (u == 0) {
    return async_leaf1(ctx);
  } else if (u == 1) {
    return async_leaf2(ctx);
  }
  return async_mutual_loop2(u - 2, ctx);
}

// CHECK: swifttailcc void @async_mutual_loop
// CHECK: tail call swifttailcc void @async_leaf1
// CHECK: tail call swifttailcc void @async_leaf2
// CHECK: tail call swifttailcc void @async_mutual_loop1
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

// CHECK-LABEL: swiftcc i8 @sync_calling_async
// CHECK-NOT: tail call
// CHECK: call swifttailcc void @async_branch
// CHECK-NOT: tail call
// CHECK: call swifttailcc void @async_loop
SWIFTCALL char sync_calling_async(_Bool b, unsigned u) {
  char x = 'a';
  async_branch(b, &x);
  async_loop(u, &x);
  return x;
}

// CHECK-LABEL: i8 @c_calling_async
// CHECK-NOT: tail call
// CHECK: call swifttailcc void @async_branch
// CHECK-NOT: tail call
// CHECK: call swifttailcc void @async_loop
char c_calling_async(_Bool b, unsigned u) {
  char x = 'a';
  async_branch(b, &x);
  async_loop(u, &x);
  return x;
}


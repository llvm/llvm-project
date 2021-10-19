// RUN: %clang_cc1 -triple=i386-pc-solaris2.11 -w -emit-llvm %s -o - | FileCheck %s

#pragma redefine_extname fake real
#pragma redefine_extname name alias

extern int fake(void);

int name;

// __PRAGMA_REDEFINE_EXTNAME should be defined.  This will fail if it isn't...
int fish() { return fake() + __PRAGMA_REDEFINE_EXTNAME + name; }
// Check that the call to fake() is emitted as a call to real()
// CHECK:   call i32 @real()
// Check that this also works with variables names
// CHECK:   load i32, i32* @alias

// This is a case when redefenition is deferred *and* we have a local of the
// same name. PR23923.
#pragma redefine_extname foo bar
int f() {
  int foo = 0;
  return foo;
}
extern int foo() { return 1; }
// CHECK: define{{.*}} i32 @bar()

// Check that pragma redefine_extname applies to external declarations only.
#pragma redefine_extname foo_static bar_static
static int foo_static() { return 1; }
int baz() { return foo_static(); }
// CHECK-NOT: call i32 @bar_static()

// Check that pragma redefine_extname applies to builtin functions.
typedef unsigned long size_t;
extern void *memcpy(void *, const void *, size_t);
#pragma redefine_extname memcpy __GI_memcpy
void *test_memcpy(void *dst, const void *src, size_t n) { return memcpy(dst, src, n); }
// CHECK: call i8* @__GI_memcpy(

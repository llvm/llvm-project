// RUN: %clang_cc1 -triple=x86_64-apple-darwin9 -emit-llvm %s -o - | FileCheck %s

extern "C" int printf(...);

static int count;

int func2(int c) { return printf("loading the func2(%d)\n", c); };
int func1(int c) { return printf("loading the func1(%d)\n", c); }

static int loader_1 = func1(++count);
// CHECK: call i32 @_Z5func1i

int loader_2 = func2(++count);

static int loader_3 = func1(++count);
// CHECK: call i32 @_Z5func1i

int main() {}

int loader_4 = func2(++count);
static int loader_5 = func1(++count);
int loader_6 = func2(++count);
// CHECK: call i32 @_Z5func1i

// CHECK-NOT: call i32 @_Z5func1i

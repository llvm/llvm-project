// RUN: %clang_cc1 -emit-llvm %s -fno-builtin -o - | FileCheck %s
// Check that -fno-builtin is honored.

extern "C" int printf(const char*, ...);
void foo(const char *msg) {
  // CHECK: call{{.*}}printf
	printf("%s\n",msg);
}

// RUN: %clang_extdef_map %s -- | FileCheck %s

int f(int) {
  return 0;
}

// CHECK: c:@F@f#I#

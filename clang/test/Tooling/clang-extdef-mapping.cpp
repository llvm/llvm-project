// RUN: clang-extdef-mapping "%s" 2>&1 | FileCheck %s

// CHECK: 8:c:@F@foo {{.*}}clang-extdef-mapping.cpp
extern "C" int foo() { return 0; }

// RUN: not %clang_cc1 %s -include "/abspath/missing file with spaces.h" 2>&1 | FileCheck %s
// CHECK: file not found
int main() { }

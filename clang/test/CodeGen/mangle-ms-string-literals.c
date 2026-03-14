// RUN: %clang_cc1 -x c -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -x c -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s

void crbug857442(int x) {
  // Make sure to handle truncated or padded literals. The truncation is only valid in C.
  struct {int x; char s[2]; } truncatedAscii = {x, "hello"};
  // CHECK: "??_C@_01CONKJJHI@he@"
  struct {int x; char s[16]; } paddedAscii = {x, "hello"};
  // CHECK: "??_C@_0BA@EAAINDNC@hello?$AA?$AA?$AA?$AA?$AA?$AA?$AA?$AA?$AA?$AA?$AA@"
}

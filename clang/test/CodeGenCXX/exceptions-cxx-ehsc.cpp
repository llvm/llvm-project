// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -fexceptions -fcxx-exceptions -fexternc-nounwind | FileCheck %s

namespace test1 {
struct Cleanup { ~Cleanup(); };
extern "C" void never_throws();
void may_throw();

void caller() {
  Cleanup x;
  never_throws();
  may_throw();
}
}
// CHECK-LABEL: define void @"\01?caller@test1@@YAXXZ"(
// CHECK: call void @never_throws(
// CHECK: invoke void @"\01?may_throw@test1@@YAXXZ"(

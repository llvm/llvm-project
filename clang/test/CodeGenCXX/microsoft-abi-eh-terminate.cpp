// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc -mconstructor-aliases -fexceptions -fcxx-exceptions -fms-compatibility-version=18.00 | FileCheck -check-prefix=MSVC2013 %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc -mconstructor-aliases -fexceptions -fcxx-exceptions -fms-compatibility-version=19.00 | FileCheck -check-prefix=MSVC2015 %s

void may_throw();
void never_throws() noexcept(true) {
  may_throw();
}

// CHECK-LABEL: define void @"\01?never_throws@@YAXXZ"()
// CHECK-SAME:          personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
// CHECK:      invoke void @"\01?may_throw@@YAXXZ"()
// MSVC2013:      terminatepad [void ()* @"\01?terminate@@YAXXZ"]
// MSVC2015:      terminatepad [void ()* @__std_terminate]
// CHECK-NEXT: unreachable

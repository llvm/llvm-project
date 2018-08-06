// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc -mconstructor-aliases -fexceptions -fcxx-exceptions -fms-compatibility-version=18.00 | FileCheck -check-prefix=MSVC2013 -check-prefix=CHECK %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc -mconstructor-aliases -fexceptions -fcxx-exceptions -fms-compatibility-version=19.00 | FileCheck -check-prefix=MSVC2015 -check-prefix=CHECK %s

void may_throw();
void never_throws() noexcept(true) {
  may_throw();
}

// CHECK-LABEL: define dso_local void @"?never_throws@@YAXXZ"()
// CHECK-SAME:          personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
// CHECK:      invoke void @"?may_throw@@YAXXZ"()
// CHECK:      %[[cp:.*]] = cleanuppad within none []
// MSVC2013:      call void @"?terminate@@YAXXZ"()
// MSVC2015:      call void @__std_terminate()
// CHECK-SAME:  [ "funclet"(token %[[cp]]) ]
// CHECK-NEXT: unreachable

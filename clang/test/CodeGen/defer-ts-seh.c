// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c23 -fdefer-ts -fms-compatibility -emit-llvm %s -o - | FileCheck %s

void g();
void h();

void f() {
  __try {
    _Defer h();
    g();
  } __finally {

  }
}

// CHECK-LABEL: define {{.*}} void @f() {{.*}} personality ptr @__C_specific_handler
// CHECK: entry:
// CHECK:   invoke void @g() #4
// CHECK:           to label %invoke.cont unwind label %ehcleanup
// CHECK: invoke.cont:
// CHECK:   invoke void @h() #4
// CHECK:           to label %invoke.cont1 unwind label %ehcleanup3
// CHECK: invoke.cont1:
// CHECK:   %0 = call ptr @llvm.localaddress()
// CHECK:   call void @"?fin$0@0@f@@"(i8 {{.*}} 0, ptr {{.*}} %0)
// CHECK:   ret void
// CHECK: ehcleanup:
// CHECK:   %1 = cleanuppad within none []
// CHECK:   invoke void @h() #4 [ "funclet"(token %1) ]
// CHECK:           to label %invoke.cont2 unwind label %ehcleanup3
// CHECK: invoke.cont2:
// CHECK:   cleanupret from %1 unwind label %ehcleanup3
// CHECK: ehcleanup3:
// CHECK:   %2 = cleanuppad within none []
// CHECK:   %3 = call ptr @llvm.localaddress()
// CHECK:   call void @"?fin$0@0@f@@"(i8 {{.*}} 1, ptr {{.*}} %3) [ "funclet"(token %2) ]
// CHECK:   cleanupret from %2 unwind to caller

// CHECK-LABEL: define {{.*}} void @"?fin$0@0@f@@"(i8 {{.*}} %abnormal_termination, ptr {{.*}} %frame_pointer)
// CHECK: entry:
// CHECK:   %frame_pointer.addr = alloca ptr, align 8
// CHECK:   %abnormal_termination.addr = alloca i8, align 1
// CHECK:   store ptr %frame_pointer, ptr %frame_pointer.addr, align 8
// CHECK:   store i8 %abnormal_termination, ptr %abnormal_termination.addr, align 1
// CHECK:   ret void

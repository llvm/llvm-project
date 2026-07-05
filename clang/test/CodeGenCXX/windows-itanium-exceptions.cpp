// RUN: %clang_cc1 -emit-llvm -triple thumbv7-windows-itanium -fexceptions -fcxx-exceptions %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple i686-windows-itanium -fexceptions -fcxx-exceptions %s -o - | FileCheck %s
// REQUIRES: asserts

void except() {
  throw 32;
}

void attempt() {
  try { except(); } catch (...) { }
}

// CHECK: @_ZTIi = external dso_local constant ptr

// CHECK: define {{.*}}void @_Z6exceptv() {{.*}} {
// CHECK:   %exception = call {{.*}}ptr @__cxa_allocate_exception(i32 4)
// CHECK:   store i32 32, ptr %exception
// CHECK:   call {{.*}}void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
// CHECK:   unreachable
// CHECK: }

// CHECK: define {{.*}}void @_Z7attemptv()
// CHECK-SAME: personality ptr @__gxx_personality_v0
// CHECK:   %exn.slot = alloca ptr
// CHECK:   %ehselector.slot = alloca i32
// CHECK:   invoke {{.*}}void @_Z6exceptv()
// CHECK:     to label %invoke.cont unwind label %lpad
// CHECK: invoke.cont:
// CHECK:    br label %try.cont
// CHECK: lpad:
// CHECK:    %0 = landingpad { ptr, i32 }
// CHECK:      catch ptr null
// CHECK:    %1 = extractvalue { ptr, i32 } %0, 0
// CHECK:    store ptr %1, ptr %exn.slot
// CHECK:    %2 = extractvalue { ptr, i32 } %0, 1
// CHECK:    store i32 %2, ptr %ehselector.slot
// CHECK:    br label %catch
// CHECK: catch:
// CHECK:    %exn = load ptr, ptr %exn.slot
// CHECK:    %3 = call {{.*}}ptr @__cxa_begin_catch(ptr %{{2|exn}})
// CHECK:    call {{.*}}void @__cxa_end_catch()
// CHECK:    br label %try.cont
// CHECK: try.cont:
// CHECK:    ret void
// CHECK: }



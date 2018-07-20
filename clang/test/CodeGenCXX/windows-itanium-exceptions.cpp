// RUN: %clang_cc1 -emit-llvm -triple thumbv7-windows-itanium -fexceptions -fcxx-exceptions %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple i686-windows-itanium -fexceptions -fcxx-exceptions %s -o - | FileCheck %s
// REQUIRES: asserts

void except() {
  throw 32;
}

void attempt() {
  try { except(); } catch (...) { }
}

// CHECK: @_ZTIi = external dso_local constant i8*

// CHECK: define {{.*}}void @_Z6exceptv() {{.*}} {
// CHECK:   %exception = call {{.*}}i8* @__cxa_allocate_exception(i32 4)
// CHECK:   %0 = bitcast i8* %exception to i32*
// CHECK:   store i32 32, i32* %0
// CHECK:   call {{.*}}void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
// CHECK:   unreachable
// CHECK: }

// CHECK: define {{.*}}void @_Z7attemptv()
// CHECK-SAME: personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
// CHECK:   %exn.slot = alloca i8*
// CHECK:   %ehselector.slot = alloca i32
// CHECK:   invoke {{.*}}void @_Z6exceptv()
// CHECK:     to label %invoke.cont unwind label %lpad
// CHECK: invoke.cont:
// CHECK:    br label %try.cont
// CHECK: lpad:
// CHECK:    %0 = landingpad { i8*, i32 }
// CHECK:      catch i8* null
// CHECK:    %1 = extractvalue { i8*, i32 } %0, 0
// CHECK:    store i8* %1, i8** %exn.slot
// CHECK:    %2 = extractvalue { i8*, i32 } %0, 1
// CHECK:    store i32 %2, i32* %ehselector.slot
// CHECK:    br label %catch
// CHECK: catch:
// CHECK:    %exn = load i8*, i8** %exn.slot
// CHECK:    %3 = call {{.*}}i8* @__cxa_begin_catch(i8* %{{2|exn}})
// CHECK:    call {{.*}}void @__cxa_end_catch()
// CHECK:    br label %try.cont
// CHECK: try.cont:
// CHECK:    ret void
// CHECK: }



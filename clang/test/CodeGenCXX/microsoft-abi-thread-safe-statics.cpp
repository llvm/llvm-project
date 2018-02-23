// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -fms-extensions -fms-compatibility -fms-compatibility-version=19 -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s
// REQUIRES: asserts

struct S {
  S();
  ~S();
};

// CHECK-DAG: @"\01?s@?1??f@@YAAAUS@@XZ@4U2@A" = linkonce_odr dso_local thread_local global %struct.S zeroinitializer
// CHECK-DAG: @"\01??__J?1??f@@YAAAUS@@XZ@51" = linkonce_odr thread_local global i32 0
// CHECK-DAG: @"\01?s@?1??g@@YAAAUS@@XZ@4U2@A" = linkonce_odr dso_local global %struct.S zeroinitializer
// CHECK-DAG: @"\01?$TSS0@?1??g@@YAAAUS@@XZ@4HA" = linkonce_odr global i32 0
// CHECK-DAG: @_Init_thread_epoch = external thread_local global i32, align 4
// CHECK-DAG: @"\01?j@?1??h@@YAAAUS@@_N@Z@4U2@A" = linkonce_odr dso_local thread_local global %struct.S zeroinitializer
// CHECK-DAG: @"\01??__J?1??h@@YAAAUS@@_N@Z@51" = linkonce_odr thread_local global i32 0
// CHECK-DAG: @"\01?i@?1??h@@YAAAUS@@_N@Z@4U2@A" = linkonce_odr dso_local global %struct.S zeroinitializer
// CHECK-DAG: @"\01?$TSS0@?1??h@@YAAAUS@@_N@Z@4HA" = linkonce_odr global i32 0
// CHECK-DAG: @"\01?i@?1??g1@@YAHXZ@4HA" = internal global i32 0, align 4
// CHECK-DAG: @"\01?$TSS0@?1??g1@@YAHXZ@4HA" = internal global i32 0, align 4

// CHECK-LABEL: define {{.*}} @"\01?f@@YAAAUS@@XZ"()
// CHECK-SAME:  personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
extern inline S &f() {
  static thread_local S s;
// CHECK:       %[[guard:.*]] = load i32, i32* @"\01??__J?1??f@@YAAAUS@@XZ@51"
// CHECK-NEXT:  %[[mask:.*]] = and i32 %[[guard]], 1
// CHECK-NEXT:  %[[cmp:.*]] = icmp eq i32 %[[mask]], 0
// CHECK-NEXT:  br i1 %[[cmp]], label %[[init:.*]], label %[[init_end:.*]], !prof ![[unlikely_threadlocal:.*]]
//
// CHECK:     [[init]]:
// CHECK-NEXT:  %[[or:.*]] = or i32 %[[guard]], 1
// CHECK-NEXT:  store i32 %[[or]], i32* @"\01??__J?1??f@@YAAAUS@@XZ@51"
// CHECK-NEXT:  invoke {{.*}} @"\01??0S@@QAE@XZ"(%struct.S* @"\01?s@?1??f@@YAAAUS@@XZ@4U2@A")
// CHECK-NEXT:    to label %[[invoke_cont:.*]] unwind label %[[lpad:.*]]
//
// CHECK:     [[invoke_cont]]:
// CHECK-NEXT:  call i32 @__tlregdtor(void ()* @"\01??__Fs@?1??f@@YAAAUS@@XZ@YAXXZ")
// CHECK-NEXT:  br label %[[init_end:.*]]

// CHECK:     [[init_end]]:
// CHECK-NEXT:  ret %struct.S* @"\01?s@?1??f@@YAAAUS@@XZ@4U2@A"

// CHECK:     [[lpad:.*]]:
// CHECK-NEXT: cleanuppad within none []
// CHECK:       %[[guard:.*]] = load i32, i32* @"\01??__J?1??f@@YAAAUS@@XZ@51"
// CHECK-NEXT:  %[[mask:.*]] = and i32 %[[guard]], -2
// CHECK-NEXT:  store i32 %[[mask]], i32* @"\01??__J?1??f@@YAAAUS@@XZ@51"
// CHECK-NEXT:  cleanupret {{.*}} unwind to caller
  return s;
}


// CHECK-LABEL: define {{.*}} @"\01?g@@YAAAUS@@XZ"()
extern inline S &g() {
  static S s;
// CHECK:  %[[guard:.*]] = load atomic i32, i32* @"\01?$TSS0@?1??g@@YAAAUS@@XZ@4HA" unordered, align 4
// CHECK-NEXT:  %[[epoch:.*]] = load i32, i32* @_Init_thread_epoch
// CHECK-NEXT:  %[[cmp:.*]] = icmp sgt i32 %[[guard]], %[[epoch]]
// CHECK-NEXT:  br i1 %[[cmp]], label %[[init_attempt:.*]], label %[[init_end:.*]], !prof ![[unlikely_staticlocal:.*]]
//
// CHECK:     [[init_attempt]]:
// CHECK-NEXT:  call void @_Init_thread_header(i32* @"\01?$TSS0@?1??g@@YAAAUS@@XZ@4HA")
// CHECK-NEXT:  %[[guard2:.*]] = load atomic i32, i32* @"\01?$TSS0@?1??g@@YAAAUS@@XZ@4HA" unordered, align 4
// CHECK-NEXT:  %[[cmp2:.*]] = icmp eq i32 %[[guard2]], -1
// CHECK-NEXT:  br i1 %[[cmp2]], label %[[init:.*]], label %[[init_end:.*]]
//
// CHECK:     [[init]]:
// CHECK-NEXT:  invoke {{.*}} @"\01??0S@@QAE@XZ"(%struct.S* @"\01?s@?1??g@@YAAAUS@@XZ@4U2@A")
// CHECK-NEXT:    to label %[[invoke_cont:.*]] unwind label %[[lpad:.*]]
//
// CHECK:     [[invoke_cont]]:
// CHECK-NEXT:  call i32 @atexit(void ()* @"\01??__Fs@?1??g@@YAAAUS@@XZ@YAXXZ")
// CHECK-NEXT:  call void @_Init_thread_footer(i32* @"\01?$TSS0@?1??g@@YAAAUS@@XZ@4HA")
// CHECK-NEXT:  br label %init.end
//
// CHECK:     [[init_end]]:
// CHECK-NEXT:  ret %struct.S* @"\01?s@?1??g@@YAAAUS@@XZ@4U2@A"
//
// CHECK:     [[lpad]]:
// CHECK-NEXT: cleanuppad within none []
// CHECK:       call void @_Init_thread_abort(i32* @"\01?$TSS0@?1??g@@YAAAUS@@XZ@4HA")
// CHECK-NEXT:  cleanupret {{.*}} unwind to caller
  return s;
}

extern inline S&h(bool b) {
  static thread_local S j;
  static S i;
  return b ? j : i;
}

// CHECK-LABEL: define dso_local i32 @"\01?g1@@YAHXZ"()
int f1();
int g1() {
  static int i = f1();
  return i;
}

// CHECK-DAG: ![[unlikely_threadlocal]] = !{!"branch_weights", i32 1, i32 1023}
// CHECK-DAG: ![[unlikely_staticlocal]] = !{!"branch_weights", i32 1, i32 1048575}

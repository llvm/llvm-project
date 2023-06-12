// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -O2 -disable-llvm-passes -o - %s | FileCheck %s

struct NSFastEnumerationState;
@interface NSArray
- (unsigned long) countByEnumeratingWithState: (struct NSFastEnumerationState*) state
                  objects: (id*) buffer
                  count: (unsigned long) bufferSize;
@end;
NSArray *nsarray() { return 0; }
// CHECK: define{{.*}} ptr @_Z7nsarrayv()

void use(id);

// rdar://problem/9315552
// The analogous ObjC testcase test46 in arr.m.
void test0(__weak id *wp, __weak volatile id *wvp) {
  extern id test0_helper(void);

  // TODO: this is sub-optimal, we should retain at the actual call site.
  // TODO: in the non-volatile case, we do not need to be reloading.

  // CHECK:      [[T1:%.*]] = call noundef ptr @_Z12test0_helperv() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr {{%.*}}, align 8
  // CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.storeWeak(ptr [[T2]], ptr [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = call ptr @llvm.objc.retain(ptr [[T3]])
  // CHECK-NEXT: store ptr [[T4]], ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  id x = *wp = test0_helper();

  // CHECK:      [[T1:%.*]] = call noundef ptr @_Z12test0_helperv() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T1]])
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr {{%.*}}, align 8
  // CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.storeWeak(ptr [[T2]], ptr [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[T2]])
  // CHECK-NEXT: store ptr [[T4]], ptr
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T1]])
  id y = *wvp = test0_helper();
}

// rdar://problem/9320648
struct Test1_helper { Test1_helper(); };
@interface Test1 @end
@implementation Test1 { Test1_helper x; } @end
// CHECK: define internal noundef ptr @"\01-[Test1 .cxx_construct]"(
// CHECK:      call void @_ZN12Test1_helperC1Ev(
// CHECK-NEXT: load
// CHECK-NEXT: ret ptr

void test34(int cond) {
  __strong id strong;
  __weak id weak;
  extern void test34_sink(id *);
  test34_sink(cond ? &strong : 0);
  test34_sink(cond ? &weak : 0);

  // CHECK-LABEL:    define{{.*}} void @_Z6test34i(
  // CHECK:      [[COND:%.*]] = alloca i32
  // CHECK-NEXT: [[STRONG:%.*]] = alloca ptr
  // CHECK-NEXT: [[WEAK:%.*]] = alloca ptr
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca ptr
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca ptr
  // CHECK-NEXT: [[CONDCLEANUPSAVE:%.*]] = alloca ptr
  // CHECK-NEXT: [[CONDCLEANUP:%.*]] = alloca i1
  // CHECK-NEXT: store i32
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[STRONG]])
  // CHECK-NEXT: store ptr null, ptr [[STRONG]]
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[WEAK]])
  // CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[WEAK]], ptr null)

  // CHECK-NEXT: [[T0:%.*]] = load i32, ptr [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi ptr
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], ptr null, ptr [[TEMP1]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[ARG]]
  // CHECK-NEXT: store ptr [[T0]], ptr [[TEMP1]]
  // CHECK-NEXT: br label
  // CHECK:      [[W0:%.*]] = phi ptr [ [[T0]], {{%.*}} ], [ undef, {{%.*}} ]
  // CHECK:      call void @_Z11test34_sinkPU15__autoreleasingP11objc_object(ptr noundef [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[TEMP1]]
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.use(ptr [[W0]])
  // CHECK-NEXT: [[T2:%.*]] = load ptr, ptr [[ARG]]
  // CHECK-NEXT: store ptr [[T1]], ptr [[ARG]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T2]])
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i32, ptr [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi ptr
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], ptr null, ptr [[TEMP2]]
  // CHECK-NEXT: store i1 false, ptr [[CONDCLEANUP]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[ARG]])
  // CHECK-NEXT: store ptr [[T0]], ptr [[CONDCLEANUPSAVE]]
  // CHECK-NEXT: store i1 true, ptr [[CONDCLEANUP]]
  // CHECK-NEXT: store ptr [[T0]], ptr [[TEMP2]]
  // CHECK-NEXT: br label
  // CHECK:      call void @_Z11test34_sinkPU15__autoreleasingP11objc_object(ptr noundef [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load ptr, ptr [[TEMP2]]
  // CHECK-NEXT: call ptr @llvm.objc.storeWeak(ptr [[ARG]], ptr [[T0]])
  // CHECK-NEXT: br label

  // CHECK:      call void @llvm.objc.destroyWeak(ptr [[WEAK]])
  // CHECK:      ret void
}

struct Test35_Helper {
  static id makeObject1() __attribute__((ns_returns_retained));
  id makeObject2() __attribute__((ns_returns_retained));
  static id makeObject3();
  id makeObject4();
};

// CHECK-LABEL: define{{.*}} void @_Z6test3513Test35_HelperPS_
void test35(Test35_Helper x0, Test35_Helper *x0p) {
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr @_ZN13Test35_Helper11makeObject1Ev
  // CHECK-NOT: call ptr @llvm.objc.retain
  id obj1 = Test35_Helper::makeObject1();
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr @_ZN13Test35_Helper11makeObject2Ev
  // CHECK-NOT: call ptr @llvm.objc.retain
  id obj2 = x0.makeObject2();
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr @_ZN13Test35_Helper11makeObject2Ev
  // CHECK-NOT: call ptr @llvm.objc.retain
  id obj3 = x0p->makeObject2();
  id (Test35_Helper::*pmf)() __attribute__((ns_returns_retained))
    = &Test35_Helper::makeObject2;
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr %
  // CHECK-NOT: call ptr @llvm.objc.retain
  id obj4 = (x0.*pmf)();
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr %
  // CHECK-NOT: call ptr @llvm.objc.retain
  id obj5 = (x0p->*pmf)();

  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z7test35b13Test35_HelperPS_
void test35b(Test35_Helper x0, Test35_Helper *x0p) {
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr @_ZN13Test35_Helper11makeObject3Ev{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  id obj1 = Test35_Helper::makeObject3();
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr @_ZN13Test35_Helper11makeObject4Ev{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  id obj2 = x0.makeObject4();
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr @_ZN13Test35_Helper11makeObject4Ev{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  id obj3 = x0p->makeObject4();
  id (Test35_Helper::*pmf)() = &Test35_Helper::makeObject4;
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr %{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  id obj4 = (x0.*pmf)();
  // CHECK: call void @llvm.lifetime.start
  // CHECK: call noundef ptr %{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  id obj5 = (x0p->*pmf)();

  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK: call void @llvm.objc.release
  // CHECK: call void @llvm.lifetime.end
  // CHECK-NEXT: ret void
}

// rdar://problem/9603128
// CHECK-LABEL: define{{.*}} ptr @_Z6test36P11objc_object(
id test36(id z) {
  // CHECK: llvm.objc.retain
  // CHECK: llvm.objc.retain
  // CHECK: llvm.objc.release
  // CHECK: llvm.objc.autoreleaseReturnValue
  return z;
}

// Template instantiation side of rdar://problem/9817306
@interface Test37
+ alloc;
- init;
- (NSArray *) array;
@end
template <class T> void test37(T *a) {
  for (id x in a.array) {
    use(x);
  }
}
extern template void test37<Test37>(Test37 *a);
template void test37<Test37>(Test37 *a);
// CHECK-LABEL: define weak_odr void @_Z6test37I6Test37EvPT_(
// CHECK:      [[T2:%.*]] = call noundef ptr @objc_msgSend({{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T2]])

// Make sure it's not immediately released before starting the iteration.
// CHECK-NEXT: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: @objc_msgSend

// This bitcast is for the mutation check.
// CHECK: @objc_enumerationMutation

// This bitcast is for the 'next' message send.
// CHECK: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: @objc_msgSend

// This bitcast is for the final release.
// CHECK: call void @llvm.objc.release(ptr [[T2]])

template<typename T>
void send_release() {
  [Test37 array];
}

// CHECK-LABEL: define weak_odr void @_Z12send_releaseIiEvv(
// CHECK: call noundef ptr @objc_msgSend{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(
// CHECK-NEXT: call void @llvm.objc.release
// CHECK-NEXT: ret void
template void send_release<int>();

template<typename T>
Test37 *instantiate_init() {
  Test37 *result = [[Test37 alloc] init];
  return result;
}

// CHECK-LABEL: define weak_odr noundef ptr @_Z16instantiate_initIiEP6Test37v
// CHECK: call noundef ptr @objc_msgSend
// CHECK: call noundef ptr @objc_msgSend
// CHECK: call ptr @llvm.objc.retain
// CHECK: call void @llvm.objc.release
// CHECK: call ptr @llvm.objc.autoreleaseReturnValue
template Test37* instantiate_init<int>();

// Just make sure that the AST invariants hold properly here,
// i.e. that we don't crash.
// The block should get bound in the full-expression outside
// the statement-expression.
template <class T> class Test38 {
  void test(T x) {
    ^{ (void) x; }, ({ x; });
  }
};
// CHECK-LABEL: define weak_odr void @_ZN6Test38IiE4testEi(
template class Test38<int>;

// rdar://problem/11964832
class Test39_base1 {
  virtual void foo();
};
class Test39_base2 {
  virtual id bar();
};
class Test39 : Test39_base1, Test39_base2 { // base2 is at non-zero offset
  virtual id bar();
};
id Test39::bar() { return 0; }
// Note lack of autorelease.
// CHECK-LABEL:    define{{.*}} ptr @_ZThn8_N6Test393barEv(
// CHECK:      call noundef ptr @_ZN6Test393barEv(
// CHECK-NEXT: ret ptr

// rdar://13617051
// Just a basic correctness check that IR-gen still works after instantiating
// a non-dependent message send that requires writeback.
@interface Test40
+ (void) foo:(id *)errorPtr;
@end
template <class T> void test40_helper() {
  id x;
  [Test40 foo: &x];
};
template void test40_helper<int>();
// CHECK-LABEL:    define weak_odr void @_Z13test40_helperIiEvv()
// CHECK:      [[X:%.*]] = alloca ptr
// CHECK-NEXT: [[TEMP:%.*]] = alloca ptr
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
// CHECK-NEXT: store ptr null, ptr [[X]]
// CHECK:      [[T0:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT: store ptr [[T0]], ptr [[TEMP]]
// CHECK:      @objc_msgSend
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[TEMP]]
// CHECK-NEXT: call ptr @llvm.objc.retain(ptr [[T0]])

// Check that moves out of __weak variables are compiled to use objc_moveWeak.
void test41(__weak id &&x) {
  __weak id y = static_cast<__weak id &&>(x);
}
// CHECK-LABEL: define{{.*}} void @_Z6test41OU6__weakP11objc_object
// CHECK:      [[X:%.*]] = alloca ptr
// CHECK:      [[Y:%.*]] = alloca ptr
// CHECK:      [[T0:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT: call void @llvm.objc.moveWeak(ptr [[Y]], ptr [[T0]])
// CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[Y]])

void test42() {
  __attribute__((ns_returns_retained)) id test42_0();
  id test42_1(id);
  void test42_2(id &&);
  test42_2(test42_1(test42_0()));
}

// Check that the pointer returned by test42_0 is released after the full expression.

// CHECK-LABEL: define void @_Z6test42v()
// CHECK: %[[CALL:.*]] = call noundef ptr @_Z8test42_0v()
// CHECK: call void @_Z8test42_2OU15__autoreleasingP11objc_object(
// CHECK: call void @llvm.objc.release(ptr %[[CALL]])

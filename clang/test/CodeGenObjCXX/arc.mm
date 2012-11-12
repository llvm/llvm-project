// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -O2 -disable-llvm-optzns -o - %s | FileCheck %s

struct NSFastEnumerationState;
@interface NSArray
- (unsigned long) countByEnumeratingWithState: (struct NSFastEnumerationState*) state
                  objects: (id*) buffer
                  count: (unsigned long) bufferSize;
@end;
NSArray *nsarray() { return 0; }
// CHECK: define [[NSARRAY:%.*]]* @_Z7nsarrayv()

void use(id);

// rdar://problem/9315552
// The analogous ObjC testcase test46 in arr.m.
void test0(__weak id *wp, __weak volatile id *wvp) {
  extern id test0_helper(void);

  // TODO: this is sub-optimal, we should retain at the actual call site.
  // TODO: in the non-volatile case, we do not need to be reloading.

  // CHECK:      [[T0:%.*]] = call i8* @_Z12test0_helperv()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8*** {{%.*}}, align 8
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_storeWeak(i8** [[T2]], i8* [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = call i8* @objc_retain(i8* [[T3]])
  // CHECK-NEXT: store i8* [[T4]], i8**
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  id x = *wp = test0_helper();

  // CHECK:      [[T0:%.*]] = call i8* @_Z12test0_helperv()
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8*** {{%.*}}, align 8
  // CHECK-NEXT: [[T3:%.*]] = call i8* @objc_storeWeak(i8** [[T2]], i8* [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = call i8* @objc_loadWeakRetained(i8** [[T2]])
  // CHECK-NEXT: store i8* [[T4]], i8**
  // CHECK-NEXT: call void @objc_release(i8* [[T1]])
  id y = *wvp = test0_helper();
}

// rdar://problem/9320648
struct Test1_helper { Test1_helper(); };
@interface Test1 @end
@implementation Test1 { Test1_helper x; } @end
// CHECK: define internal i8* @"\01-[Test1 .cxx_construct]"(
// CHECK:      call void @_ZN12Test1_helperC1Ev(
// CHECK-NEXT: load
// CHECK-NEXT: bitcast
// CHECK-NEXT: ret i8*

void test34(int cond) {
  __strong id strong;
  __weak id weak;
  extern void test34_sink(id *);
  test34_sink(cond ? &strong : 0);
  test34_sink(cond ? &weak : 0);

  // CHECK:    define void @_Z6test34i(
  // CHECK:      [[COND:%.*]] = alloca i32
  // CHECK-NEXT: [[STRONG:%.*]] = alloca i8*
  // CHECK-NEXT: [[WEAK:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP1:%.*]] = alloca i8*
  // CHECK-NEXT: [[TEMP2:%.*]] = alloca i8*
  // CHECK-NEXT: store i32
  // CHECK-NEXT: store i8* null, i8** [[STRONG]]
  // CHECK-NEXT: call i8* @objc_initWeak(i8** [[WEAK]], i8* null)

  // CHECK-NEXT: [[T0:%.*]] = load i32* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi i8**
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], i8** null, i8** [[TEMP1]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8** [[ARG]]
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP1]]
  // CHECK-NEXT: br label
  // CHECK:      call void @_Z11test34_sinkPU15__autoreleasingP11objc_object(i8** [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8** [[TEMP1]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: [[T2:%.*]] = load i8** [[ARG]]
  // CHECK-NEXT: store i8* [[T1]], i8** [[ARG]]
  // CHECK-NEXT: call void @objc_release(i8* [[T2]])
  // CHECK-NEXT: br label

  // CHECK:      [[T0:%.*]] = load i32* [[COND]]
  // CHECK-NEXT: [[T1:%.*]] = icmp ne i32 [[T0]], 0
  // CHECK:      [[ARG:%.*]] = phi i8**
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: [[T1:%.*]] = select i1 [[T0]], i8** null, i8** [[TEMP2]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = call i8* @objc_loadWeak(i8** [[ARG]])
  // CHECK-NEXT: store i8* [[T0]], i8** [[TEMP2]]
  // CHECK-NEXT: br label
  // CHECK:      call void @_Z11test34_sinkPU15__autoreleasingP11objc_object(i8** [[T1]])
  // CHECK-NEXT: [[T0:%.*]] = icmp eq i8** [[ARG]], null
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[T0:%.*]] = load i8** [[TEMP2]]
  // CHECK-NEXT: call i8* @objc_storeWeak(i8** [[ARG]], i8* [[T0]])
  // CHECK-NEXT: br label

  // CHECK:      call void @objc_destroyWeak(i8** [[WEAK]])
  // CHECK:      ret void
}

struct Test35_Helper {
  static id makeObject1() __attribute__((ns_returns_retained));
  id makeObject2() __attribute__((ns_returns_retained));
  static id makeObject3();
  id makeObject4();
};

// CHECK: define void @_Z6test3513Test35_HelperPS_
void test35(Test35_Helper x0, Test35_Helper *x0p) {
  // CHECK: call i8* @_ZN13Test35_Helper11makeObject1Ev
  // CHECK-NOT: call i8* @objc_retain
  id obj1 = Test35_Helper::makeObject1();
  // CHECK: call i8* @_ZN13Test35_Helper11makeObject2Ev
  // CHECK-NOT: call i8* @objc_retain
  id obj2 = x0.makeObject2();
  // CHECK: call i8* @_ZN13Test35_Helper11makeObject2Ev
  // CHECK-NOT: call i8* @objc_retain
  id obj3 = x0p->makeObject2();
  id (Test35_Helper::*pmf)() __attribute__((ns_returns_retained))
    = &Test35_Helper::makeObject2;
  // CHECK: call i8* %
  // CHECK-NOT: call i8* @objc_retain
  id obj4 = (x0.*pmf)();
  // CHECK: call i8* %
  // CHECK-NOT: call i8* @objc_retain
  id obj5 = (x0p->*pmf)();

  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void
}

// CHECK: define void @_Z7test35b13Test35_HelperPS_
void test35b(Test35_Helper x0, Test35_Helper *x0p) {
  // CHECK: call i8* @_ZN13Test35_Helper11makeObject3Ev
  // CHECK: call i8* @objc_retain
  id obj1 = Test35_Helper::makeObject3();
  // CHECK: call i8* @_ZN13Test35_Helper11makeObject4Ev
  // CHECK: call i8* @objc_retain
  id obj2 = x0.makeObject4();
  // CHECK: call i8* @_ZN13Test35_Helper11makeObject4Ev
  // CHECK: call i8* @objc_retain
  id obj3 = x0p->makeObject4();
  id (Test35_Helper::*pmf)() = &Test35_Helper::makeObject4;
  // CHECK: call i8* %
  // CHECK: call i8* @objc_retain
  id obj4 = (x0.*pmf)();
  // CHECK: call i8* %
  // CHECK: call i8* @objc_retain
  id obj5 = (x0p->*pmf)();

  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void
}

// rdar://problem/9603128
// CHECK: define i8* @_Z6test36P11objc_object(
id test36(id z) {
  // CHECK: objc_retain
  // CHECK: objc_retain
  // CHECK: objc_release
  // CHECK: objc_autoreleaseReturnValue
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
// CHECK: define weak_odr void @_Z6test37I6Test37EvPT_(
// CHECK:      [[T0:%.*]] = call [[NSARRAY]]* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to [[NSARRAY]]* (i8*, i8*)*)(
// CHECK-NEXT: [[T1:%.*]] = bitcast [[NSARRAY]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT: [[COLL:%.*]] = bitcast i8* [[T2]] to [[NSARRAY]]*

// Make sure it's not immediately released before starting the iteration.
// CHECK-NEXT: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: [[T0:%.*]] = bitcast [[NSARRAY]]* [[COLL]] to i8*
// CHECK-NEXT: @objc_msgSend

// This bitcast is for the mutation check.
// CHECK:      [[T0:%.*]] = bitcast [[NSARRAY]]* [[COLL]] to i8*
// CHECK-NEXT: @objc_enumerationMutation

// This bitcast is for the 'next' message send.
// CHECK:      [[T0:%.*]] = bitcast [[NSARRAY]]* [[COLL]] to i8*
// CHECK-NEXT: @objc_msgSend

// This bitcast is for the final release.
// CHECK:      [[T0:%.*]] = bitcast [[NSARRAY]]* [[COLL]] to i8*
// CHECK-NEXT: call void @objc_release(i8* [[T0]])

template<typename T>
void send_release() {
  [Test37 array];
}

// CHECK: define weak_odr void @_Z12send_releaseIiEvv(
// CHECK: call %0* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK-NEXT: bitcast
// CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
// CHECK-NEXT: bitcast
// CHECK-NEXT: bitcast
// CHECK-NEXT: call void @objc_release
// CHECK-NEXT: ret void
template void send_release<int>();

template<typename T>
Test37 *instantiate_init() {
  Test37 *result = [[Test37 alloc] init];
  return result;
}

// CHECK: define weak_odr %2* @_Z16instantiate_initIiEP6Test37v
// CHECK: call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: call i8* @objc_retain
// CHECK: call void @objc_release
// CHECK: call i8* @objc_autoreleaseReturnValue
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
// CHECK: define weak_odr void @_ZN6Test38IiE4testEi(
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
// CHECK:    define i8* @_ZThn8_N6Test393barEv(
// CHECK:      call i8* @_ZN6Test393barEv(
// CHECK-NEXT: ret i8*

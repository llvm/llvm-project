// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O1 -fno-split-cold-code -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O1 -fno-split-cold-code -fcxx-exceptions -fexceptions -std=c++03 -o - %s | FileCheck --check-prefixes=CHECK-EH,CHECK-EH-03 %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O1 -fno-split-cold-code -fcxx-exceptions -fexceptions -std=c++11 -o - %s | FileCheck --check-prefixes=CHECK-EH,CHECK-EH-11 %s

// Test code generation for the named return value optimization.
class X {
public:
  X();
  X(const X&);
  ~X();
};

template<typename T> struct Y {
  Y();
  static Y f() {
    Y y;
    return y;
  }
};

// CHECK-LABEL: define void @_Z5test0v
// CHECK-EH-LABEL: define void @_Z5test0v
X test0() {
  X x;
  // CHECK:          call {{.*}} @_ZN1XC1Ev
  // CHECK-NEXT:     ret void

  // CHECK-EH:       call {{.*}} @_ZN1XC1Ev
  // CHECK-EH-NEXT:  ret void
  return x;
}

// CHECK-LABEL: define void @_Z5test1b(
// CHECK-EH-LABEL: define void @_Z5test1b(
X test1(bool B) {
  // CHECK:      tail call {{.*}} @_ZN1XC1Ev
  // CHECK-NEXT: ret void
  X x;
  if (B)
    return (x);
  return x;
  // CHECK-EH:      tail call {{.*}} @_ZN1XC1Ev
  // CHECK-EH-NEXT: ret void
}

// CHECK-LABEL: define void @_Z5test2b
// CHECK-EH-LABEL: define void @_Z5test2b
// CHECK-EH-SAME:  personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
X test2(bool B) {
  // No NRVO.

  X x;
  X y;
  if (B)
    return y;
  return x;

  // CHECK: call {{.*}} @_ZN1XC1Ev
  // CHECK-NEXT: {{.*}} getelementptr inbounds %class.X, %class.X* %y, i32 0, i32 0
  // CHECK-NEXT: call void @llvm.lifetime.start
  // CHECK-NEXT: call {{.*}} @_ZN1XC1Ev
  // CHECK: call {{.*}} @_ZN1XC1ERKS_
  // CHECK: call {{.*}} @_ZN1XC1ERKS_
  // CHECK: call {{.*}} @_ZN1XD1Ev
  // CHECK-NEXT: call void @llvm.lifetime.end
  // CHECK: call {{.*}} @_ZN1XD1Ev
  // CHECK-NEXT: call void @llvm.lifetime.end
  // CHECK: ret void

  // The block ordering in the -fexceptions IR is unfortunate.

  // CHECK-EH:      call void @llvm.lifetime.start
  // CHECK-EH-NEXT: call {{.*}} @_ZN1XC1Ev
  // CHECK-EH:      call void @llvm.lifetime.start
  // CHECK-EH-NEXT: invoke {{.*}} @_ZN1XC1Ev
  // -> %invoke.cont, %lpad

  // %invoke.cont:
  // CHECK-EH:      br i1
  // -> %if.then, %if.end

  // %if.then: returning 'x'
  // CHECK-EH:      invoke {{.*}} @_ZN1XC1ERKS_
  // -> %cleanup, %lpad1

  // %lpad: landing pad for ctor of 'y', dtor of 'y'
  // CHECK-EH:      [[CAUGHTVAL:%.*]] = landingpad { i8*, i32 }
  // CHECK-EH-NEXT:   cleanup
  // CHECK-EH-NEXT: extractvalue { i8*, i32 } [[CAUGHTVAL]], 0
  // CHECK-EH-NEXT: extractvalue { i8*, i32 } [[CAUGHTVAL]], 1
  // CHECK-EH-NEXT: br label
  // -> %eh.cleanup

  // %lpad1: landing pad for return copy ctors, EH cleanup for 'y'
  // CHECK-EH-03: invoke {{.*}} @_ZN1XD1Ev
  // -> %eh.cleanup, %terminate.lpad
  // CHECK-EH-11: call   {{.*}} @_ZN1XD1Ev

  // %if.end: returning 'y'
  // CHECK-EH: invoke {{.*}} @_ZN1XC1ERKS_
  // -> %cleanup, %lpad1

  // %cleanup: normal cleanup for 'y'
  // CHECK-EH-03: invoke {{.*}} @_ZN1XD1Ev
  // -> %invoke.cont11, %lpad
  // CHECK-EH-11: call   {{.*}} @_ZN1XD1Ev

  // %invoke.cont11: normal cleanup for 'x'
  // CHECK-EH:      call void @llvm.lifetime.end
  // CHECK-EH-NEXT: call {{.*}} @_ZN1XD1Ev
  // CHECK-EH-NEXT: call void @llvm.lifetime.end
  // CHECK-EH-NEXT: ret void

  // %eh.cleanup:  EH cleanup for 'x'
  // CHECK-EH-03: invoke {{.*}} @_ZN1XD1Ev
  // -> %invoke.cont17, %terminate.lpad
  // CHECK-EH-11: call   {{.*}} @_ZN1XD1Ev

  // %invoke.cont17: rethrow block for %eh.cleanup.
  // This really should be elsewhere in the function.
  // CHECK-EH:      resume { i8*, i32 }

  // %terminate.lpad: terminate landing pad.
  // CHECK-EH-03:      [[T0:%.*]] = landingpad { i8*, i32 }
  // CHECK-EH-03-NEXT:   catch i8* null
  // CHECK-EH-03-NEXT: [[T1:%.*]] = extractvalue { i8*, i32 } [[T0]], 0
  // CHECK-EH-03-NEXT: call void @__clang_call_terminate(i8* [[T1]]) [[NR_NUW:#[0-9]+]]
  // CHECK-EH-03-NEXT: unreachable

}

// CHECK-LABEL: define void @_Z5test3b
X test3(bool B) {
  // CHECK: tail call {{.*}} @_ZN1XC1Ev
  // CHECK-NOT: call {{.*}} @_ZN1XC1ERKS_
  // CHECK: call {{.*}} @_ZN1XC1Ev
  // CHECK: call {{.*}} @_ZN1XC1ERKS_
  if (B) {
    X y;
    return y;
  }
  // FIXME: we should NRVO this variable too.
  X x;
  return x;
}

extern "C" void exit(int) throw();

// CHECK-LABEL: define void @_Z5test4b
X test4(bool B) {
  {
    // CHECK: tail call {{.*}} @_ZN1XC1Ev
    X x;
    // CHECK: br i1
    if (B)
      return x;
  }
  // CHECK: tail call {{.*}} @_ZN1XD1Ev
  // CHECK: tail call void @exit(i32 1)
  exit(1);
}

#ifdef __EXCEPTIONS
// CHECK-EH-LABEL: define void @_Z5test5
void may_throw();
X test5() {
  try {
    may_throw();
  } catch (X x) {
    // CHECK-EH: invoke {{.*}} @_ZN1XC1ERKS_
    // CHECK-EH: call void @__cxa_end_catch()
    // CHECK-EH: ret void
    return x;
  }
}
#endif

// rdar://problem/10430868
// CHECK-LABEL: define void @_Z5test6v
X test6() {
  X a __attribute__((aligned(8)));
  return a;
  // CHECK:      [[A:%.*]] = alloca [[X:%.*]], align 8
  // CHECK-NEXT: [[PTR:%.*]] = getelementptr inbounds %class.X, %class.X* [[A]], i32 0, i32 0
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull [[PTR]])
  // CHECK-NEXT: call {{.*}} @_ZN1XC1Ev([[X]]* nonnull [[A]])
  // CHECK-NEXT: call {{.*}} @_ZN1XC1ERKS_([[X]]* {{%.*}}, [[X]]* nonnull dereferenceable({{[0-9]+}}) [[A]])
  // CHECK-NEXT: call {{.*}} @_ZN1XD1Ev([[X]]* nonnull [[A]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull [[PTR]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define void @_Z5test7b
X test7(bool b) {
  // CHECK: tail call {{.*}} @_ZN1XC1Ev
  // CHECK-NEXT: ret
  if (b) {
    X x;
    return x;
  }
  return X();
}

// CHECK-LABEL: define void @_Z5test8b
X test8(bool b) {
  // CHECK: tail call {{.*}} @_ZN1XC1Ev
  // CHECK-NEXT: ret
  if (b) {
    X x;
    return x;
  } else {
    X y;
    return y;
  }
}

Y<int> test9() {
  Y<int>::f();
}

// CHECK-LABEL: define linkonce_odr void @_ZN1YIiE1fEv
// CHECK: tail call {{.*}} @_ZN1YIiEC1Ev

// CHECK-EH-03: attributes [[NR_NUW]] = { noreturn nounwind }

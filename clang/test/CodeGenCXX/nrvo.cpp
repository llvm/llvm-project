// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O1 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O1 -fcxx-exceptions -fexceptions -o - %s | FileCheck --check-prefix=CHECK-EH %s

// Test code generation for the named return value optimization.
class X {
public:
  X();
  X(const X&);
  ~X();
};

// CHECK: define void @_Z5test0v
// CHECK-EH: define void @_Z5test0v
X test0() {
  X x;
  // CHECK:          call {{.*}} @_ZN1XC1Ev
  // CHECK-NEXT:     ret void

  // CHECK-EH:       call {{.*}} @_ZN1XC1Ev
  // CHECK-EH-NEXT:  ret void
  return x;
}

// CHECK: define void @_Z5test1b(
// CHECK-EH: define void @_Z5test1b(
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

// CHECK: define void @_Z5test2b
// CHECK-EH: define void @_Z5test2b
X test2(bool B) {
  // No NRVO.

  X x;
  X y;
  if (B)
    return y;
  return x;

  // CHECK: call {{.*}} @_ZN1XC1Ev
  // CHECK-NEXT: call {{.*}} @_ZN1XC1Ev
  // CHECK: call {{.*}} @_ZN1XC1ERKS_
  // CHECK: call {{.*}} @_ZN1XC1ERKS_
  // CHECK: call {{.*}} @_ZN1XD1Ev
  // CHECK: call {{.*}} @_ZN1XD1Ev
  // CHECK: ret void

  // The block ordering in the -fexceptions IR is unfortunate.

  // CHECK-EH:      call {{.*}} @_ZN1XC1Ev
  // CHECK-EH-NEXT: invoke {{.*}} @_ZN1XC1Ev
  // -> %invoke.cont, %lpad

  // %invoke.cont:
  // CHECK-EH:      br i1
  // -> %if.then, %if.end

  // %if.then: returning 'x'
  // CHECK-EH:      invoke {{.*}} @_ZN1XC1ERKS_
  // -> %cleanup, %lpad1

  // %lpad: landing pad for ctor of 'y', dtor of 'y'
  // CHECK-EH:      [[CAUGHTVAL:%.*]] = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-EH-NEXT:   cleanup
  // CHECK-EH-NEXT: extractvalue { i8*, i32 } [[CAUGHTVAL]], 0
  // CHECK-EH-NEXT: extractvalue { i8*, i32 } [[CAUGHTVAL]], 1
  // CHECK-EH-NEXT: br label
  // -> %eh.cleanup

  // %lpad1: landing pad for return copy ctors, EH cleanup for 'y'
  // CHECK-EH: invoke {{.*}} @_ZN1XD1Ev
  // -> %eh.cleanup, %terminate.lpad

  // %if.end: returning 'y'
  // CHECK-EH: invoke {{.*}} @_ZN1XC1ERKS_
  // -> %cleanup, %lpad1

  // %cleanup: normal cleanup for 'y'
  // CHECK-EH: invoke {{.*}} @_ZN1XD1Ev
  // -> %invoke.cont11, %lpad

  // %invoke.cont11: normal cleanup for 'x'
  // CHECK-EH:      call {{.*}} @_ZN1XD1Ev
  // CHECK-EH-NEXT: ret void

  // %eh.cleanup:  EH cleanup for 'x'
  // CHECK-EH: invoke {{.*}} @_ZN1XD1Ev
  // -> %invoke.cont17, %terminate.lpad

  // %invoke.cont17: rethrow block for %eh.cleanup.
  // This really should be elsewhere in the function.
  // CHECK-EH:      resume { i8*, i32 }

  // %terminate.lpad: terminate landing pad.
  // CHECK-EH:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-EH-NEXT:   catch i8* null
  // CHECK-EH-NEXT: call void @_ZSt9terminatev()
  // CHECK-EH-NEXT: unreachable

}

X test3(bool B) {
  // FIXME: We don't manage to apply NRVO here, although we could.
  {
    X y;
    return y;
  }
  X x;
  return x;
}

extern "C" void exit(int) throw();

// CHECK: define void @_Z5test4b
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
// CHECK-EH: define void @_Z5test5
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
// CHECK: define void @_Z5test6v
X test6() {
  X a __attribute__((aligned(8)));
  return a;
  // CHECK:      [[A:%.*]] = alloca [[X:%.*]], align 8
  // CHECK-NEXT: call {{.*}} @_ZN1XC1Ev([[X]]* [[A]])
  // CHECK-NEXT: call {{.*}} @_ZN1XC1ERKS_([[X]]* {{%.*}}, [[X]]* [[A]])
  // CHECK-NEXT: call {{.*}} @_ZN1XD1Ev([[X]]* [[A]])
  // CHECK-NEXT: ret void
}

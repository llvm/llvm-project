// RUN: %clang_cc1 -triple x86_64-darwin -std=c++11 -fobjc-arc -emit-llvm -o - %s | FileCheck %s --implicit-check-not "call\ "

struct Strong {
  __strong id x;
};

struct Base {
  // Use variadic args to cause inlining the inherited constructor.
  Base(Strong s, ...) {}
};

struct NonTrivialDtor {
  ~NonTrivialDtor() {}
};
struct Inheritor : public NonTrivialDtor, public Base {
  using Base::Base;
};

id g(void);
void f() {
  Inheritor({g()});
}
// CHECK-LABEL: define{{.*}} void @_Z1fv
// CHECK:       %call1 = call noundef ptr @_Z1gv() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK:       call void (...) @llvm.objc.clang.arc.noop.use(ptr %call1) #2
// CHECK:       call void (ptr, ptr, ...) @_ZN4BaseC2E6Strongz(ptr {{.*}}, ptr {{.*}})
// CHECK-NEXT:  call void @_ZN9InheritorD1Ev(ptr {{.*}})

// CHECK-LABEL: define linkonce_odr void @_ZN4BaseC2E6Strongz(ptr {{.*}}, ptr {{.*}}, ...)
// CHECK:       call void @_ZN6StrongD1Ev(ptr {{.*}})

// CHECK-LABEL: define linkonce_odr void @_ZN9InheritorD1Ev(ptr {{.*}})
// CHECK:       call void @_ZN9InheritorD2Ev(ptr {{.*}})

// CHECK-LABEL: define linkonce_odr void @_ZN6StrongD1Ev(ptr {{.*}})
// CHECK:       call void @_ZN6StrongD2Ev(ptr {{.*}})

// CHECK-LABEL: define linkonce_odr void @_ZN6StrongD2Ev(ptr {{.*}})
// CHECK:       call void @llvm.objc.storeStrong(ptr {{.*}}, ptr null)

// CHECK-LABEL: define linkonce_odr void @_ZN9InheritorD2Ev(ptr {{.*}})
// CHECK:       call void @_ZN14NonTrivialDtorD2Ev(ptr {{.*}})

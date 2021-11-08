// RUN: %clang_cc1 -triple x86_64-darwin -std=c++11 -fobjc-arc -emit-llvm -o - %s | FileCheck %s --implicit-check-not "call\ "
// rdar://problem/45805151

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
// CHECK:       %[[TMP:.*]] = call i8* @_Z1gv()
// CHECK:       {{.*}} = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[TMP]])
// CHECK:       call void (%struct.Base*, i8*, ...) @_ZN4BaseC2E6Strongz(%struct.Base* {{.*}}, i8* {{.*}})
// CHECK-NEXT:  call void @_ZN9InheritorD1Ev(%struct.Inheritor* {{.*}})

// CHECK-LABEL: define linkonce_odr void @_ZN4BaseC2E6Strongz(%struct.Base* {{.*}}, i8* {{.*}}, ...)
// CHECK:       call void @_ZN6StrongD1Ev(%struct.Strong* {{.*}})

// CHECK-LABEL: define linkonce_odr void @_ZN9InheritorD1Ev(%struct.Inheritor* {{.*}})
// CHECK:       call void @_ZN9InheritorD2Ev(%struct.Inheritor* {{.*}})

// CHECK-LABEL: define linkonce_odr void @_ZN6StrongD1Ev(%struct.Strong* {{.*}})
// CHECK:       call void @_ZN6StrongD2Ev(%struct.Strong* {{.*}})

// CHECK-LABEL: define linkonce_odr void @_ZN6StrongD2Ev(%struct.Strong* {{.*}})
// CHECK:       call void @llvm.objc.storeStrong(i8** {{.*}}, i8* null)

// CHECK-LABEL: define linkonce_odr void @_ZN9InheritorD2Ev(%struct.Inheritor* {{.*}})
// CHECK:       call void @_ZN14NonTrivialDtorD2Ev(%struct.NonTrivialDtor* {{.*}})

// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -o - %s | FileCheck %s

// Copy constructor
struct X0 {
  X0();
  X0(const X0 &) throw();
  X0(X0 &);
};

struct X1 {
  X1();
  X1(const X1 &) throw();
};

struct X2 : X1 { 
  X2();
};
struct X3 : X0, X1 { 
  X3();
};

struct X4 {
  X4(X4 &) throw();
};

struct X5 : X0, X4 { };

void test(X2 x2, X3 x3, X5 x5) {
  // CHECK: define linkonce_odr void @_ZN2X2C1ERKS_(%struct.X2* {{[^,]*}} %this, %struct.X2* noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0) unnamed_addr
  // CHECK:      call void @_ZN2X2C2ERKS_({{.*}}) [[NUW:#[0-9]+]]
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  X2 x2a(x2);
  // CHECK: define linkonce_odr void @_ZN2X3C1ERKS_(%struct.X3* {{[^,]*}} %this, %struct.X3* noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0) unnamed_addr
  // CHECK:      call void @_ZN2X3C2ERKS_({{.*}}) [[NUW]]
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
  X3 x3a(x3);
  // CHECK: define linkonce_odr void @_ZN2X5C1ERS_({{.*}}) unnamed_addr
  // CHECK-NOT: call void @__cxa_call_unexpected
  // CHECK: ret void
  X5 x5a(x5);
}

// Default constructor
struct X6 {
  X6() throw();
};

struct X7 { 
  X7();
};

struct X8 : X6 { };
struct X9 : X6, X7 { };

void test() {
  // CHECK: define linkonce_odr void @_ZN2X8C1Ev(%struct.X8* {{[^,]*}} %this) unnamed_addr
  // CHECK:      call void @_ZN2X8C2Ev({{.*}}) [[NUW]]
  // CHECK-NEXT: ret void
  X8();

  // CHECK: define linkonce_odr void @_ZN2X9C1Ev(%struct.X9* {{[^,]*}} %this) unnamed_addr
  //   FIXME: check that this is the end of the line here:
  // CHECK:      call void @_ZN2X9C2Ev({{.*}})
  // CHECK-NEXT: ret void
  X9();

  // CHECK: define linkonce_odr void @_ZN2X8C2Ev(%struct.X8* {{[^,]*}} %this) unnamed_addr
  // CHECK:      call void @_ZN2X6C2Ev({{.*}}) [[NUW]]
  // CHECK-NEXT: ret void

  // CHECK: define linkonce_odr void @_ZN2X9C2Ev(%struct.X9* {{[^,]*}} %this) unnamed_addr
  // CHECK:      call void @_ZN2X6C2Ev({{.*}}) [[NUW]]
  //   FIXME: and here:
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @_ZN2X7C2Ev({{.*}})
  // CHECK: ret void
}

// CHECK: attributes [[NUW]] = { nounwind{{.*}} }

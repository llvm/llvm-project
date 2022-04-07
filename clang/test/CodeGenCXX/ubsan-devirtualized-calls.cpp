// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -triple %itanium_abi_triple -emit-llvm -fsanitize=null,vptr %s -o - | FileCheck %s

struct Base1 {
  virtual void f1() {}
};

struct Base2 {
  virtual void f1() {}
};

struct Derived1 final : Base1 {
  void f1() override {}
};

struct Derived2 final : Base1, Base2 {
  void f1() override {}
};

struct Derived3 : Base1 {
  void f1() override /* nofinal */ {}
};

struct Derived4 final : Base1 {
  void f1() override final {}
};

// CHECK: [[UBSAN_TI_DERIVED1_1:@[0-9]+]] = private unnamed_addr global {{.*}} i8* bitcast {{.*}} @_ZTI8Derived1 to i8*
// CHECK: [[UBSAN_TI_DERIVED2_1:@[0-9]+]] = private unnamed_addr global {{.*}} i8* bitcast {{.*}} @_ZTI8Derived2 to i8*
// CHECK: [[UBSAN_TI_DERIVED2_2:@[0-9]+]] = private unnamed_addr global {{.*}} i8* bitcast {{.*}} @_ZTI8Derived2 to i8*
// CHECK: [[UBSAN_TI_DERIVED3:@[0-9]+]] = private unnamed_addr global {{.*}} i8* bitcast {{.*}} @_ZTI8Derived3 to i8*
// CHECK: [[UBSAN_TI_BASE1:@[0-9]+]] = private unnamed_addr global {{.*}} i8* bitcast {{.*}} @_ZTI5Base1 to i8*
// CHECK: [[UBSAN_TI_DERIVED4_1:@[0-9]+]] = private unnamed_addr global {{.*}} i8* bitcast {{.*}} @_ZTI8Derived4 to i8*
// CHECK: [[UBSAN_TI_DERIVED4_2:@[0-9]+]] = private unnamed_addr global {{.*}} i8* bitcast {{.*}} @_ZTI8Derived4 to i8*

// CHECK-LABEL: define {{(dso_local )?}}void @_Z2t1v
void t1() {
  Derived1 d1;
  static_cast<Base1 *>(&d1)->f1(); //< Devirt Base1::f1 to Derived1::f1.
  // CHECK: %[[D1:[0-9]+]] = ptrtoint %struct.Derived1* %d1 to i{{[0-9]+}}, !nosanitize
  // CHECK-NEXT: call void @__ubsan_handle_dynamic_type_cache{{[_a-z]*}}({{.*}} [[UBSAN_TI_DERIVED1_1]] {{.*}}, i{{[0-9]+}} %[[D1]]
}

// CHECK-LABEL: define {{(dso_local )?}}void @_Z2t2v
void t2() {
  Derived2 d2;
  static_cast<Base1 *>(&d2)->f1(); //< Devirt Base1::f1 to Derived2::f1.
  // CHECK: %[[D2_1:[0-9]+]] = ptrtoint %struct.Derived2* %d2 to i{{[0-9]+}}, !nosanitize
  // CHECK-NEXT: call void @__ubsan_handle_dynamic_type_cache{{[_a-z]*}}({{.*}} [[UBSAN_TI_DERIVED2_1]] {{.*}}, i{{[0-9]+}} %[[D2_1]]
}

// CHECK-LABEL: define {{(dso_local )?}}void @_Z2t3v
void t3() {
  Derived2 d2;
  static_cast<Base2 *>(&d2)->f1(); //< Devirt Base2::f1 to Derived2::f1.
  // CHECK: %[[D2_2:[0-9]+]] = ptrtoint %struct.Derived2* %d2 to i{{[0-9]+}}, !nosanitize
  // CHECK-NEXT: call void @__ubsan_handle_dynamic_type_cache{{[_a-z]*}}({{.*}} [[UBSAN_TI_DERIVED2_2]] {{.*}}, i{{[0-9]+}} %[[D2_2]]
}

// CHECK-LABEL: define {{(dso_local )?}}void @_Z2t4v
void t4() {
  Base1 p;
  Derived3 *badp = static_cast<Derived3 *>(&p); //< Check that &p isa Derived3.
  // CHECK: %[[P1:[0-9]+]] = ptrtoint %struct.Derived3* {{%[0-9]+}} to i{{[0-9]+}}, !nosanitize
  // CHECK-NEXT: call void @__ubsan_handle_dynamic_type_cache{{[_a-z]*}}({{.*}} [[UBSAN_TI_DERIVED3]] {{.*}}, i{{[0-9]+}} %[[P1]]

  static_cast<Base1 *>(badp)->f1(); //< No devirt, test 'badp isa Base1'.
  // We were able to skip the null check on the first type check because 'p'
  // is backed by an alloca. We can't skip the second null check because 'badp'
  // is a (bitcast (load ...)).
  // CHECK: call void @__ubsan_handle_type_mismatch
  //
  // CHECK: %[[BADP1:[0-9]+]] = ptrtoint %struct.Base1* {{%[0-9]+}} to i{{[0-9]+}}, !nosanitize
  // CHECK-NEXT: call void @__ubsan_handle_dynamic_type_cache{{[_a-z]*}}({{.*}} [[UBSAN_TI_BASE1]] {{.*}}, i{{[0-9]+}} %[[BADP1]]
}

// CHECK-LABEL: define {{(dso_local )?}}void @_Z2t5v
void t5() {
  Base1 p;
  Derived4 *badp = static_cast<Derived4 *>(&p); //< Check that &p isa Derived4.
  // CHECK: %[[P1:[0-9]+]] = ptrtoint %struct.Derived4* {{%[0-9]+}} to i{{[0-9]+}}, !nosanitize
  // CHECK-NEXT: call void @__ubsan_handle_dynamic_type_cache{{[_a-z]*}}({{.*}} [[UBSAN_TI_DERIVED4_1]] {{.*}}, i{{[0-9]+}} %[[P1]]

  static_cast<Base1 *>(badp)->f1(); //< Devirt Base1::f1 to Derived4::f1.
  // CHECK: call void @__ubsan_handle_type_mismatch
  //
  // CHECK: %[[BADP1:[0-9]+]] = ptrtoint %struct.Derived4* {{%[0-9]+}} to i{{[0-9]+}}, !nosanitize
  // CHECK-NEXT: call void @__ubsan_handle_dynamic_type_cache{{[_a-z]*}}({{.*}} [[UBSAN_TI_DERIVED4_2]] {{.*}}, i{{[0-9]+}} %[[BADP1]]
}

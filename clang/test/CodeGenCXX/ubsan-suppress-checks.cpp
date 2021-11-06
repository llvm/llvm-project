// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=alignment | FileCheck %s --check-prefixes=CHECK,ALIGN
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=null | FileCheck %s --check-prefixes=CHECK,NULL
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=alignment,null -DCHECK_LAMBDA | FileCheck %s --check-prefixes=LAMBDA

// CHECK-LABEL: define{{.*}} void @_Z22load_non_null_pointersv
void load_non_null_pointers() {
  int var;
  var = *&var;

  int arr[1];
  arr[0] = arr[0];

  char c = "foo"[0];

  // CHECK-NOT: and i64 {{.*}}, !nosanitize
  // CHECK-NOT: icmp ne {{.*}}, null, !nosanitize
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z31use_us16_aligned_array_elementsv
void use_us16_aligned_array_elements() {
  static const unsigned short Arr[] = {0, 1, 2};
  auto use_array = [](const unsigned short(&X)[3]) -> void {};
  use_array(Arr);

  // CHECK-NOT: br i1 true
  // ALIGN-NOT: call void @__ubsan_handle_type_mismatch
  // CHECK: ret void
}

struct A {
  int foo;

  // CHECK-LABEL: define linkonce_odr void @_ZN1A10do_nothingEv
  void do_nothing() {
    // ALIGN: %[[THISINT1:[0-9]+]] = ptrtoint %struct.A* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT1]], 3, !nosanitize
    // NULL: icmp ne %struct.A* %[[THIS1:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.A* %[[THIS1]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    // CHECK: ret void
  }

#ifdef CHECK_LAMBDA
  // LAMBDA-LABEL: define linkonce_odr void @_ZN1A22do_nothing_with_lambdaEv
  void do_nothing_with_lambda() {
    // LAMBDA: icmp ne %struct.A* %[[THIS2:[a-z0-9]+]], null, !nosanitize
    // LAMBDA: %[[THISINT2:[0-9]+]] = ptrtoint %struct.A* %[[THIS2]] to i64, !nosanitize
    // LAMBDA: and i64 %[[THISINT2]], 3, !nosanitize
    // LAMBDA: call void @__ubsan_handle_type_mismatch

    auto f = [&] {
      foo = 0;
    };
    f();

    // LAMBDA-NOT: call void @__ubsan_handle_type_mismatch
    // LAMBDA: ret void
  }

// Check the IR for the lambda:
//
// LAMBDA-LABEL: define linkonce_odr void @_ZZN1A22do_nothing_with_lambdaEvENKUlvE_clEv
// LAMBDA: call void @__ubsan_handle_type_mismatch
// LAMBDA-NOT: call void @__ubsan_handle_type_mismatch
// LAMBDA: ret void
#endif

  // CHECK-LABEL: define linkonce_odr i32 @_ZN1A11load_memberEv
  int load_member() {
    // ALIGN: %[[THISINT3:[0-9]+]] = ptrtoint %struct.A* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT3]], 3, !nosanitize
    // NULL: icmp ne %struct.A* %[[THIS3:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.A* %[[THIS3]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    return foo;
    // CHECK: ret i32
  }

  // CHECK-LABEL: define linkonce_odr i32 @_ZN1A11call_methodEv
  int call_method() {
    // ALIGN: %[[THISINT4:[0-9]+]] = ptrtoint %struct.A* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT4]], 3, !nosanitize
    // NULL: icmp ne %struct.A* %[[THIS4:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.A* %[[THIS4]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    return load_member();
    // CHECK: ret i32
  }

  // CHECK-LABEL: define linkonce_odr void @_ZN1A15assign_member_1Ev
  void assign_member_1() {
    // ALIGN: %[[THISINT5:[0-9]+]] = ptrtoint %struct.A* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT5]], 3, !nosanitize
    // NULL: icmp ne %struct.A* %[[THIS5:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.A* %[[THIS5]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    foo = 0;
    // CHECK: ret void
  }

  // CHECK-LABEL: define linkonce_odr void @_ZN1A15assign_member_2Ev
  void assign_member_2() {
    // ALIGN: %[[THISINT6:[0-9]+]] = ptrtoint %struct.A* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT6]], 3, !nosanitize
    // NULL: icmp ne %struct.A* %[[THIS6:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.A* %[[THIS6]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    (__extension__ (this))->foo = 0;
    // CHECK: ret void
  }

  // CHECK-LABEL: define linkonce_odr void @_ZNK1A15assign_member_3Ev
  void assign_member_3() const {
    // ALIGN: %[[THISINT7:[0-9]+]] = ptrtoint %struct.A* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT7]], 3, !nosanitize
    // NULL: icmp ne %struct.A* %[[THIS7:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.A* %[[THIS7]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    const_cast<A *>(this)->foo = 0;
    // CHECK: ret void
  }

  // CHECK-LABEL: define linkonce_odr i32 @_ZN1A22call_through_referenceERS_
  static int call_through_reference(A &a) {
    // ALIGN: %[[OBJINT:[0-9]+]] = ptrtoint %struct.A* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[OBJINT]], 3, !nosanitize
    // ALIGN: call void @__ubsan_handle_type_mismatch
    // NULL-NOT: call void @__ubsan_handle_type_mismatch
    return a.load_member();
    // CHECK: ret i32
  }

  // CHECK-LABEL: define linkonce_odr i32 @_ZN1A20call_through_pointerEPS_
  static int call_through_pointer(A *a) {
    // CHECK: call void @__ubsan_handle_type_mismatch
    return a->load_member();
    // CHECK: ret i32
  }
};

struct B {
  operator A*() const { return nullptr; }

  // CHECK-LABEL: define linkonce_odr i32 @_ZN1B11load_memberEPS_
  static int load_member(B *bp) {
    // Check &b before converting it to an A*.
    // CHECK: call void @__ubsan_handle_type_mismatch
    //
    // Check the result of the conversion before using it.
    // NULL: call void @__ubsan_handle_type_mismatch
    //
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    return static_cast<A *>(*bp)->load_member();
    // CHECK: ret i32
  }
};

struct Base {
  int foo;

  virtual int load_member_1() = 0;
};

struct Derived : public Base {
  int bar;

  // CHECK-LABEL: define linkonce_odr i32 @_ZN7Derived13load_member_2Ev
  int load_member_2() {
    // ALIGN: %[[THISINT8:[0-9]+]] = ptrtoint %struct.Derived* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT8]], 7, !nosanitize
    // ALIGN: call void @__ubsan_handle_type_mismatch
    // NULL: icmp ne %struct.Derived* %[[THIS8:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.Derived* %[[THIS8]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    //
    // Check the result of the cast before using it.
    // CHECK: call void @__ubsan_handle_type_mismatch
    //
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    return dynamic_cast<Base *>(this)->load_member_1();
    // CHECK: ret i32
  }

  // CHECK-LABEL: define linkonce_odr i32 @_ZN7Derived13load_member_3Ev
  int load_member_3() {
    // ALIGN: %[[THISINT9:[0-9]+]] = ptrtoint %struct.Derived* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT9]], 7, !nosanitize
    // ALIGN: call void @__ubsan_handle_type_mismatch
    // ALIGN: call void @__ubsan_handle_type_mismatch
    // NULL: icmp ne %struct.Derived* %[[THIS9:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.Derived* %[[THIS9]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    return reinterpret_cast<Derived *>(static_cast<Base *>(this))->foo;
    // CHECK: ret i32
  }

  // CHECK-LABEL: define linkonce_odr i32 @_ZN7Derived13load_member_1Ev
  int load_member_1() override {
    // ALIGN: %[[THISINT10:[0-9]+]] = ptrtoint %struct.Derived* %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %[[THISINT10]], 7, !nosanitize
    // ALIGN: call void @__ubsan_handle_type_mismatch
    // NULL: icmp ne %struct.Derived* %[[THIS10:[a-z0-9]+]], null, !nosanitize
    // NULL: ptrtoint %struct.Derived* %[[THIS10]] to i64, !nosanitize
    // CHECK: call void @__ubsan_handle_type_mismatch
    // CHECK-NOT: call void @__ubsan_handle_type_mismatch
    return foo + bar;
    // CHECK: ret i32
  }
};

void force_irgen() {
  A *a;
  a->do_nothing();
#ifdef CHECK_LAMBDA
  a->do_nothing_with_lambda();
#endif
  a->load_member();
  a->call_method();
  a->assign_member_1();
  a->assign_member_2();
  a->assign_member_3();
  A::call_through_reference(*a);
  A::call_through_pointer(a);

  B::load_member(nullptr);

  Base *b = new Derived;
  b->load_member_1();

  Derived *d;
  d->load_member_2();
  d->load_member_3();

  load_non_null_pointers();
  use_us16_aligned_array_elements();
}

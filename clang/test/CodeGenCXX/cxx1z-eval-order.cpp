// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -o - -triple %itanium_abi_triple | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ITANIUM
// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -o - -triple i686-windows | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-WINDOWS
// RUN: %clang_cc1 -std=c++1z %s -emit-llvm -o - -triple x86_64-windows | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-WINDOWS
// RUN: %clang_cc1 -std=c++23 %s -emit-llvm -o - -triple %itanium_abi_triple | FileCheck %s --check-prefix=CXX23 --check-prefix=CXX23-ITANIUM
// RUN: %clang_cc1 -std=c++23 %s -emit-llvm -o - -triple i686-windows | FileCheck %s --check-prefix=CXX23 --check-prefix=CXX23-WINDOWS
// RUN: %clang_cc1 -std=c++23 %s -emit-llvm -o - -triple x86_64-windows | FileCheck %s --check-prefix=CXX23 --check-prefix=CXX23-WINDOWS

struct B;
struct A {
  A();
  A(const A&);

  void operator[](B b);

  int a_member_f(B);
};
struct B {
  B();
  ~B();
};

struct C {
  operator int *();
  A *operator->();
  void operator->*(A);
  friend void operator->*(C, B);

  friend void operator<<(C, B);
  friend void operator>>(C, B);
  void operator<<(A);
  void operator>>(A);

  void operator=(A);
  void operator+=(A);
  friend void operator+=(C, B);

  void operator,(A);
  friend void operator,(C, B);

  void operator&&(A);
  void operator||(A);
  friend void operator&&(C, B);
  friend void operator||(C, B);
};

A make_a();
A *make_a_ptr();
int A::*make_mem_ptr_a();
void (A::*make_mem_fn_ptr_a())();
B make_b();
C make_c();
void side_effect();

void callee(A);
void (*get_f())(A);


// CHECK-LABEL: define {{.*}}@{{.*}}postfix_before_args{{.*}}(
void postfix_before_args() {
  // CHECK: call {{.*}}@{{.*}}get_f{{.*}}(
  // CHECK-ITANIUM: call {{.*}}@_ZN1AC1Ev(
  // CHECK-WINDOWS: call {{.*}}@"??0A@@Q{{AE|EAA}}@XZ"(
  // CHECK: call {{.*}}%{{.*}}(
  get_f()(A{});

  // CHECK: call {{.*}}@{{.*}}side_effect{{.*}}(
  // CHECK-ITANIUM: call {{.*}}@_ZN1AC1Ev(
  // CHECK-WINDOWS: call {{.*}}@"??0A@@Q{{AE|EAA}}@XZ"(
  // CHECK: call {{.*}}@{{.*}}callee{{.*}}(
  (side_effect(), callee)(A{});
// CHECK: }
}


// CHECK-LABEL: define {{.*}}@{{.*}}dot_lhs_before_rhs{{.*}}(
void dot_lhs_before_rhs() {
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  // CHECK: call {{.*}}@{{.*}}a_member_f{{.*}}(
  make_a().a_member_f(make_b());

  // CHECK: call {{.*}}@{{.*}}make_a_ptr{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  // CHECK: call {{.*}}@{{.*}}a_member_f{{.*}}(
  make_a_ptr()->a_member_f(make_b());

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  // CHECK: call {{.*}}@{{.*}}a_member_f{{.*}}(
  make_c()->a_member_f(make_b());
// CHECK: }
}


// CHECK-LABEL: define {{.*}}@{{.*}}array_lhs_before_rhs{{.*}}(
void array_lhs_before_rhs() {
  int (&get_arr())[10];
  extern int get_index();

  // CHECK: call {{.*}}@{{.*}}get_arr{{.*}}(
  // CHECK: call {{.*}}@{{.*}}get_index{{.*}}(
  get_arr()[get_index()] = 0;

  // CHECK: call {{.*}}@{{.*}}get_index{{.*}}(
  // CHECK: call {{.*}}@{{.*}}get_arr{{.*}}(
  get_index()[get_arr()] = 0;

  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  // CHECK: call
  make_a()[make_b()];

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}get_index{{.*}}(
  // CHECK: call
  make_c()[get_index()] = 0;

  // CHECK: call {{.*}}@{{.*}}get_index{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call
  get_index()[make_c()] = 0;
// CHECK: }
}


void *operator new(decltype(sizeof(0)), C);

// CHECK-LABEL: define {{.*}}@{{.*}}alloc_before_init{{.*}}(
void alloc_before_init() {
  struct Q { Q(A) {} };
  // CHECK-ITANIUM: call {{.*}}@_Znw{{.*}}(
  // CHECK-WINDOWS: call {{.*}}@"??2@YAP{{EAX_K|AXI}}@Z"(
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  delete new Q(make_a());

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  new (make_c()) Q(make_a());
// CHECK: }
}


// CHECK-LABEL: define {{.*}}@{{.*}}dotstar_lhs_before_rhs{{.*}}(
int dotstar_lhs_before_rhs() {
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_mem_ptr_a{{.*}}(
  int a = make_a().*make_mem_ptr_a();

  // CHECK: call {{.*}}@{{.*}}make_a_ptr{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_mem_ptr_a{{.*}}(
  int b = make_a_ptr()->*make_mem_ptr_a();

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  make_c()->*make_a();

  // FIXME: For MS ABI, the order of destruction of parameters here will not be
  // reverse construction order (parameters are destroyed left-to-right in the
  // callee). That sadly seems unavoidable; the rules are not implementable as
  // specified. If we changed parameter destruction order for these functions
  // to right-to-left, we could make the destruction order match for all cases
  // other than indirect calls, but we can't completely avoid the problem.
  //
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  make_c()->*make_b();

  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_mem_fn_ptr_a{{.*}}(
  // CHECK: call
  (make_a().*make_mem_fn_ptr_a())();

  // CHECK: call {{.*}}@{{.*}}make_a_ptr{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_mem_fn_ptr_a{{.*}}(
  // CHECK: call
  (make_a_ptr()->*make_mem_fn_ptr_a())();

  return a + b;
// CHECK: }
}


// CHECK-LABEL: define {{.*}}@{{.*}}assign_rhs_before_lhs{{.*}}(
void assign_rhs_before_lhs() {
  extern int &lhs_ref(), rhs();

  // CHECK: call {{.*}}@{{.*}}rhs{{.*}}(
  // CHECK: call {{.*}}@{{.*}}lhs_ref{{.*}}(
  lhs_ref() = rhs();

  // CHECK: call {{.*}}@{{.*}}rhs{{.*}}(
  // CHECK: call {{.*}}@{{.*}}lhs_ref{{.*}}(
  lhs_ref() += rhs();

  // CHECK: call {{.*}}@{{.*}}rhs{{.*}}(
  // CHECK: call {{.*}}@{{.*}}lhs_ref{{.*}}(
  lhs_ref() %= rhs();

  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  make_c() = make_a();

  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  make_c() += make_a();

  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  make_c() += make_b();
// CHECK: }
}

// CHECK-LABEL: define {{.*}}@{{.*}}shift_lhs_before_rhs{{.*}}(
void shift_lhs_before_rhs() {
  extern int lhs(), rhs();

  // CHECK: call {{.*}}@{{.*}}lhs{{.*}}(
  // CHECK: call {{.*}}@{{.*}}rhs{{.*}}(
  (void)(lhs() << rhs());

  // CHECK: call {{.*}}@{{.*}}lhs{{.*}}(
  // CHECK: call {{.*}}@{{.*}}rhs{{.*}}(
  (void)(lhs() >> rhs());

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  make_c() << make_a();

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  make_c() >> make_a();

  // FIXME: This is not correct for Windows ABIs, see above.
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  make_c() << make_b();

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  make_c() >> make_b();
// CHECK: }
}

// CHECK-LABEL: define {{.*}}@{{.*}}comma_lhs_before_rhs{{.*}}(
void comma_lhs_before_rhs() {
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  make_c() , make_a();

  // FIXME: This is not correct for Windows ABIs, see above.
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  make_c() , make_b();
}

// CHECK-LABEL: define {{.*}}@{{.*}}andor_lhs_before_rhs{{.*}}(
void andor_lhs_before_rhs() {
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  make_c() && make_a();

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_a{{.*}}(
  make_c() || make_a();

  // FIXME: This is not correct for Windows ABIs, see above.
  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  make_c() && make_b();

  // CHECK: call {{.*}}@{{.*}}make_c{{.*}}(
  // CHECK: call {{.*}}@{{.*}}make_b{{.*}}(
  make_c() || make_b();
}

// The C++23 cases live here rather than in a deducing-this test because the
// rule under test is operand order, not the explicit object parameter itself.
#if __cplusplus >= 202302L
// An operator with an explicit object parameter takes the object as an ordinary
// argument, so the object-before-rest order is not implied by the this-first
// emission path.
struct D {
  void operator[](this D self, B b);
  void operator[](this D self, B b, C c);
  void operator[](this D self, B b, C c, A a);
  void operator()(this D self, B b, C c);
};
struct E {
  static void operator()(B b, C c);
};
D make_d();
E make_e();

// One case per function, so a label bounds each order assertion.
// CXX23-LABEL: define {{.*}}@{{.*}}subscript_object_before_index{{.*}}(
void subscript_object_before_index() {
  // CXX23: call {{.*}}@{{.*}}make_d{{.*}}(
  // CXX23: call {{.*}}@{{.*}}make_b{{.*}}(
  make_d()[make_b()];
// CXX23: }
}

// CXX23-LABEL: define {{.*}}@{{.*}}subscript_object_before_indices{{.*}}(
void subscript_object_before_indices() {
  // The indices are unsequenced against each other, so they keep the ABI order.
  // CXX23: call {{.*}}@{{.*}}make_d{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_b{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_c{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_c{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_b{{.*}}(
  make_d()[make_b(), make_c()];
// CXX23: }
}

// CXX23-LABEL: define {{.*}}@{{.*}}subscript_object_before_three_indices{{.*}}(
void subscript_object_before_three_indices() {
  // CXX23: call {{.*}}@{{.*}}make_d{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_b{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_c{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_a{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_a{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_c{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_b{{.*}}(
  make_d()[make_b(), make_c(), make_a()];
// CXX23: }
}

// CXX23-LABEL: define {{.*}}@{{.*}}call_object_before_args{{.*}}(
void call_object_before_args() {
  // CXX23: call {{.*}}@{{.*}}make_d{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_b{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_c{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_c{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_b{{.*}}(
  make_d()(make_b(), make_c());
// CXX23: }
}

// CXX23-LABEL: define {{.*}}@{{.*}}static_operator_object_first{{.*}}(
void static_operator_object_first() {
  // CXX23: call {{.*}}@{{.*}}make_e{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_b{{.*}}(
  // CXX23-ITANIUM: call {{.*}}@{{.*}}make_c{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_c{{.*}}(
  // CXX23-WINDOWS: call {{.*}}@{{.*}}make_b{{.*}}(
  make_e()(make_b(), make_c());
// CXX23: }
}
#endif

// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Port of clang/test/CodeGenCXX/cxx1z-eval-order.cpp. The MS ABI RUN lines
// are not ported because ClangIR does not support the MS ABI yet.

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


// CIR-LABEL: cir.func {{.*}} @_Z19postfix_before_argsv(
// LLVM-LABEL: define {{.*}} @_Z19postfix_before_argsv(
void postfix_before_args() {
  // CIR:  cir.call @_Z5get_fv(
  // CIR:  cir.call @_ZN1AC1Ev(
  // CIR:  cir.call %{{.+}}(
  // LLVM: call {{.*}} @_Z5get_fv(
  // LLVM: call {{.*}} @_ZN1AC1Ev(
  // LLVM: call void %{{.+}}(
  get_f()(A{});

  // CIR:  cir.call @_Z11side_effectv(
  // CIR:  cir.call @_ZN1AC1Ev(
  // CIR:  cir.call @_Z6callee1A(
  // LLVM: call {{.*}} @_Z11side_effectv(
  // LLVM: call {{.*}} @_ZN1AC1Ev(
  // LLVM: call {{.*}} @_Z6callee1A(
  (side_effect(), callee)(A{});
}


// CIR-LABEL: cir.func {{.*}} @_Z18dot_lhs_before_rhsv(
// LLVM-LABEL: define {{.*}} @_Z18dot_lhs_before_rhsv(
void dot_lhs_before_rhs() {
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_ZN1A10a_member_fE1B(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_ZN1A10a_member_fE1B(
  make_a().a_member_f(make_b());

  // CIR:  cir.call @_Z10make_a_ptrv(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_ZN1A10a_member_fE1B(
  // LLVM: call {{.*}} @_Z10make_a_ptrv(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_ZN1A10a_member_fE1B(
  make_a_ptr()->a_member_f(make_b());

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_ZN1CptEv(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_ZN1A10a_member_fE1B(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_ZN1CptEv(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_ZN1A10a_member_fE1B(
  make_c()->a_member_f(make_b());
}


// CIR-LABEL: cir.func {{.*}} @_Z20array_lhs_before_rhsv(
// LLVM-LABEL: define {{.*}} @_Z20array_lhs_before_rhsv(
void array_lhs_before_rhs() {
  int (&get_arr())[10];
  extern int get_index();

  // CIR:  cir.call @_Z7get_arrv(
  // CIR:  cir.call @_Z9get_indexv(
  // LLVM: call {{.*}} @_Z7get_arrv(
  // LLVM: call {{.*}} @_Z9get_indexv(
  get_arr()[get_index()] = 0;

  // CIR:  cir.call @_Z9get_indexv(
  // CIR:  cir.call @_Z7get_arrv(
  // LLVM: call {{.*}} @_Z9get_indexv(
  // LLVM: call {{.*}} @_Z7get_arrv(
  get_index()[get_arr()] = 0;

  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_ZN1AixE1B(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_ZN1AixE1B(
  make_a()[make_b()];

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_ZN1CcvPiEv(
  // CIR:  cir.call @_Z9get_indexv(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_ZN1CcvPiEv(
  // LLVM: call {{.*}} @_Z9get_indexv(
  make_c()[get_index()] = 0;

  // CIR:  cir.call @_Z9get_indexv(
  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_ZN1CcvPiEv(
  // LLVM: call {{.*}} @_Z9get_indexv(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_ZN1CcvPiEv(
  get_index()[make_c()] = 0;
}


void *operator new(decltype(sizeof(0)), C);

// CIR-LABEL: cir.func {{.*}} @_Z17alloc_before_initv(
// LLVM-LABEL: define {{.*}} @_Z17alloc_before_initv(
void alloc_before_init() {
  struct Q { Q(A) {} };
  // CIR:  cir.call @_Znwm(
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_ZZ17alloc_before_initvEN1QC1E1A(
  // LLVM: call {{.*}} @_Znwm(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_ZZ17alloc_before_initvEN1QC1E1A(
  delete new Q(make_a());

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Znwm1C(
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_ZZ17alloc_before_initvEN1QC1E1A(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Znwm1C(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_ZZ17alloc_before_initvEN1QC1E1A(
  new (make_c()) Q(make_a());
}


// CIR-LABEL: cir.func {{.*}} @_Z22dotstar_lhs_before_rhsv(
// LLVM-LABEL: define {{.*}} @_Z22dotstar_lhs_before_rhsv(
int dotstar_lhs_before_rhs() {
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_Z14make_mem_ptr_av(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_Z14make_mem_ptr_av(
  int a = make_a().*make_mem_ptr_a();

  // CIR:  cir.call @_Z10make_a_ptrv(
  // CIR:  cir.call @_Z14make_mem_ptr_av(
  // LLVM: call {{.*}} @_Z10make_a_ptrv(
  // LLVM: call {{.*}} @_Z14make_mem_ptr_av(
  int b = make_a_ptr()->*make_mem_ptr_a();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_ZN1CpmE1A(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_ZN1CpmE1A(
  make_c()->*make_a();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_Zpm1C1B(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_Zpm1C1B(
  make_c()->*make_b();

  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_Z17make_mem_fn_ptr_av(
  // CIR:  cir.call %{{.+}}(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_Z17make_mem_fn_ptr_av(
  // LLVM: call void %{{.+}}(
  (make_a().*make_mem_fn_ptr_a())();

  // CIR:  cir.call @_Z10make_a_ptrv(
  // CIR:  cir.call @_Z17make_mem_fn_ptr_av(
  // CIR:  cir.call %{{.+}}(
  // LLVM: call {{.*}} @_Z10make_a_ptrv(
  // LLVM: call {{.*}} @_Z17make_mem_fn_ptr_av(
  // LLVM: call void %{{.+}}(
  (make_a_ptr()->*make_mem_fn_ptr_a())();

  return a + b;
}


// CIR-LABEL: cir.func {{.*}} @_Z21assign_rhs_before_lhsv(
// LLVM-LABEL: define {{.*}} @_Z21assign_rhs_before_lhsv(
void assign_rhs_before_lhs() {
  extern int &lhs_ref(), rhs();

  // CIR:  cir.call @_Z3rhsv(
  // CIR:  cir.call @_Z7lhs_refv(
  // LLVM: call {{.*}} @_Z3rhsv(
  // LLVM: call {{.*}} @_Z7lhs_refv(
  lhs_ref() = rhs();

  // CIR:  cir.call @_Z3rhsv(
  // CIR:  cir.call @_Z7lhs_refv(
  // LLVM: call {{.*}} @_Z3rhsv(
  // LLVM: call {{.*}} @_Z7lhs_refv(
  lhs_ref() += rhs();

  // CIR:  cir.call @_Z3rhsv(
  // CIR:  cir.call @_Z7lhs_refv(
  // LLVM: call {{.*}} @_Z3rhsv(
  // LLVM: call {{.*}} @_Z7lhs_refv(
  lhs_ref() %= rhs();

  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_ZN1CaSE1A(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_ZN1CaSE1A(
  make_c() = make_a();

  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_ZN1CpLE1A(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_ZN1CpLE1A(
  make_c() += make_a();

  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_ZpL1C1B(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_ZpL1C1B(
  make_c() += make_b();
}

// CIR-LABEL: cir.func {{.*}} @_Z20shift_lhs_before_rhsv(
// LLVM-LABEL: define {{.*}} @_Z20shift_lhs_before_rhsv(
void shift_lhs_before_rhs() {
  extern int lhs(), rhs();

  // CIR:  cir.call @_Z3lhsv(
  // CIR:  cir.call @_Z3rhsv(
  // LLVM: call {{.*}} @_Z3lhsv(
  // LLVM: call {{.*}} @_Z3rhsv(
  (void)(lhs() << rhs());

  // CIR:  cir.call @_Z3lhsv(
  // CIR:  cir.call @_Z3rhsv(
  // LLVM: call {{.*}} @_Z3lhsv(
  // LLVM: call {{.*}} @_Z3rhsv(
  (void)(lhs() >> rhs());

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_ZN1ClsE1A(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_ZN1ClsE1A(
  make_c() << make_a();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_ZN1CrsE1A(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_ZN1CrsE1A(
  make_c() >> make_a();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_Zls1C1B(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_Zls1C1B(
  make_c() << make_b();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_Zrs1C1B(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_Zrs1C1B(
  make_c() >> make_b();
}

// CIR-LABEL: cir.func {{.*}} @_Z20comma_lhs_before_rhsv(
// LLVM-LABEL: define {{.*}} @_Z20comma_lhs_before_rhsv(
void comma_lhs_before_rhs() {
  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_ZN1CcmE1A(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_ZN1CcmE1A(
  make_c() , make_a();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_Zcm1C1B(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_Zcm1C1B(
  make_c() , make_b();
}

// CIR-LABEL: cir.func {{.*}} @_Z20andor_lhs_before_rhsv(
// LLVM-LABEL: define {{.*}} @_Z20andor_lhs_before_rhsv(
void andor_lhs_before_rhs() {
  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_ZN1CaaE1A(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_ZN1CaaE1A(
  make_c() && make_a();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_av(
  // CIR:  cir.call @_ZN1CooE1A(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_av(
  // LLVM: call {{.*}} @_ZN1CooE1A(
  make_c() || make_a();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_Zaa1C1B(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_Zaa1C1B(
  make_c() && make_b();

  // CIR:  cir.call @_Z6make_cv(
  // CIR:  cir.call @_Z6make_bv(
  // CIR:  cir.call @_Zoo1C1B(
  // LLVM: call {{.*}} @_Z6make_cv(
  // LLVM: call {{.*}} @_Z6make_bv(
  // LLVM: call {{.*}} @_Zoo1C1B(
  make_c() || make_b();
}

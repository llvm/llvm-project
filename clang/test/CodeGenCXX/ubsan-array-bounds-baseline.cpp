// C++ counterpart of ubsan-array-bounds-baseline.c: reference binding, member
// calls, base conversions, pointers to members and default arguments. As there,
// a case named `..._arrow` is the same access as its non-arrow counterpart and
// the two must agree.
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=array-bounds \
// RUN:     -Wno-array-bounds -std=c++17 %s -o - | FileCheck %s
//
// Constructors are emitted after the free functions, so the cases whose check
// lands in one are checked under their own prefix.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=array-bounds \
// RUN:     -Wno-array-bounds -std=c++17 %s -o - | FileCheck %s \
// RUN:     --check-prefix=CTOR

struct T {
  T();
  ~T();
};
struct M {
  int f;
  void m();
};
struct Base {
  int b;
};
struct Derived : Base {
  int d;
};
struct Agg {
  int x;
};

int a[4];
int a2[4][4];
M ma[4];
Derived da[4];
Agg agga[4];
Agg agglocal;
int gidx;
int *p;
M *q;
int M::*pmf = &M::f;

// Separate from M so that adding a vptr does not perturb the cases above.
struct MS {
  int f;
  static void stat();
  virtual void virt();
};
MS msa[4];

//===----------------------------------------------------------------------===//
// Contexts that require the element to exist.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}}@_Z10x_ref_bindi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_ref_bind(int i) {
  int &r = a[i];
  (void)r;
}

// CHECK-LABEL: define {{.*}}@_Z11x_ref_consti(
// CHECK: icmp ule i64 {{.*}}, 4
void x_ref_const(int i) {
  const int &r = a[i];
  (void)r;
}

// CHECK-LABEL: define {{.*}}@_Z14x_ref_cleanupsi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_ref_cleanups(int i) {
  int &r = (T(), a[i]);
  (void)r;
}

// CHECK-LABEL: define {{.*}}@_Z12x_ref_returni(
// CHECK: icmp ule i64 {{.*}}, 4
int &x_ref_return(int i) { return a[i]; }

void takes_ref(int &);
// CHECK-LABEL: define {{.*}}@_Z11x_ref_parami(
// CHECK: icmp ule i64 {{.*}}, 4
void x_ref_param(int i) { takes_ref(a[i]); }

// CHECK-LABEL: define {{.*}}@_Z16x_ref_structuredi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_ref_structured(int i) {
  auto &[e] = agga[i];
  (void)e;
}

// A default argument bound to a reference.
int &pick(int &r = a[gidx]);
// CHECK-LABEL: define {{.*}}@_Z13x_default_argv(
// CHECK: icmp ule i64 {{.*}}, 4
void x_default_arg() { pick(); }

// A default member initializer bound to a reference. The check lands in the
// constructor.
struct R {
  int &r = a[gidx];
};
// CTOR-LABEL: define {{.*}}@_ZN1RC2Ev(
// CTOR: icmp ule i64 {{.*}}, 4
void x_default_init() {
  R x;
  (void)x;
}

struct S {
  int &r;
  S(int i) : r(a[i]) {}
};
// CTOR-LABEL: define {{.*}}@_ZN1SC2Ei(
// CTOR: icmp ule i64 {{.*}}, 4
void x_ref_meminit(int i) {
  S x(i);
  (void)x;
}

// CHECK-LABEL: define {{.*}}@_Z13x_member_calli(
// CHECK: icmp ule i64 {{.*}}, 4
void x_member_call(int i) { ma[i].m(); }

// CHECK-LABEL: define {{.*}}@_Z19x_member_call_arrowi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_member_call_arrow(int i) { (&ma[i])->m(); }

// CHECK-LABEL: define {{.*}}@_Z21x_member_call_virtuali(
// CHECK: icmp ule i64 {{.*}}, 4
void x_member_call_virtual(int i) { msa[i].virt(); }

// TODO: confirm with reviewers. C++ [class.static]p2 says the object expression
// is evaluated, but the call does not use the object, and whether evaluating a
// glvalue that designates no object is undefined when nothing reads it is not
// settled (see CWG 232). Permissive is the conservative answer; CHECK-NOT guards
// against the member-call rule over-applying to it.
// CHECK-LABEL: define {{.*}}@_Z20x_member_call_statici(
// CHECK: icmp ule i64 {{.*}}, 4
// CHECK-NOT: icmp ult
void x_member_call_static(int i) { msa[i].stat(); }

// CHECK-LABEL: define {{.*}}@_Z12x_member_doti(
// CHECK: icmp ult i64 {{.*}}, 4
void x_member_dot(int i) { ma[i].f = 1; }

// CHECK-LABEL: define {{.*}}@_Z14x_member_arrowi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_member_arrow(int i) { (&ma[i])->f = 1; }

// CHECK-LABEL: define {{.*}}@_Z20x_trivial_assign_lhsi(
// CHECK: icmp ult i64 {{.*}}, 4
void x_trivial_assign_lhs(int i) { agga[i] = agglocal; }

// CHECK-LABEL: define {{.*}}@_Z20x_trivial_assign_rhsi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_trivial_assign_rhs(int i) { agglocal = agga[i]; }

// The same assignment spelled as an explicit operator= call.
// CHECK-LABEL: define {{.*}}@_Z17x_explicit_assigni(
// CHECK: icmp ule i64 {{.*}}, 4
void x_explicit_assign(int i) { agglocal.operator=(agga[i]); }

// CHECK-LABEL: define {{.*}}@_Z17x_derived_to_basei(
// CHECK: icmp ule i64 {{.*}}, 4
void x_derived_to_base(int i) {
  Base &r = da[i];
  (void)r;
}

// CHECK-LABEL: define {{.*}}@_Z18x_static_cast_basei(
// CHECK: icmp ule i64 {{.*}}, 4
void x_static_cast_base(int i) {
  Base &r = static_cast<Base &>(da[i]);
  (void)r;
}

// The other direction.
Base ba[4];
// CHECK-LABEL: define {{.*}}@_Z17x_base_to_derivedi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_base_to_derived(int i) {
  Derived &r = static_cast<Derived &>(ba[i]);
  (void)r;
}

// CHECK-LABEL: define {{.*}}@_Z15x_ptr_to_memberi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_ptr_to_member(int i) { ma[i].*pmf = 1; }

// CHECK-LABEL: define {{.*}}@_Z21x_ptr_to_member_arrowi(
// CHECK: icmp ule i64 {{.*}}, 4
void x_ptr_to_member_arrow(int i) { (&ma[i])->*pmf = 1; }

// CHECK-LABEL: define {{.*}}@_Z10x_cond_armib(
// CHECK: icmp ult i64 {{.*}}, 4
void x_cond_arm(int i, bool c) { (c ? a[i] : a[0]) = 1; }

// An assignment is a prvalue in C and an lvalue in C++, so the four cases below
// reach a different emitter than their equivalents in the C file.
// CHECK-LABEL: define {{.*}}@_Z7x_storei(
// CHECK: icmp ult i64 {{.*}}, 4
void x_store(int i) { a[i] = 1; }

// CHECK-LABEL: define {{.*}}@_Z7x_pareni(
// CHECK: icmp ult i64 {{.*}}, 4
void x_paren(int i) { (a[i]) = 1; }

// CHECK-LABEL: define {{.*}}@_Z12x_deref_addri(
// CHECK: icmp ult i64 {{.*}}, 4
void x_deref_addr(int i) { *&a[i] = 1; }

// C has no glvalue comma, so the C file can only test the pointer
// form, c_comma_addr.
// CHECK-LABEL: define {{.*}}@_Z7x_commai(
// CHECK: icmp ult i64 {{.*}}, 4
void x_comma(int i) { (1, a[i]) = 1; }

// CHECK-LABEL: define {{.*}}@_Z11x_copy_initi(
// CHECK: icmp ult i64 {{.*}}, 4
void x_copy_init(int i) {
  Agg b = agga[i];
  (void)b;
}

// The element passed by value, through the same constructor.
void xsink(Agg);
// CHECK-LABEL: define {{.*}}@_Z7x_byvali(
// CHECK: icmp ult i64 {{.*}}, 4
void x_byval(int i) { xsink(agga[i]); }

// With a non-trivial copy constructor there is a real call and the argument binds
// to `const T &`, so this is reference binding instead.
struct NT {
  NT();
  NT(const NT &);
  int x;
};
NT nta[4];
void ntsink(NT);
// A non-trivial copy constructor makes a real call, so the
// argument binds to a reference instead.
// CHECK-LABEL: define {{.*}}@_Z18x_byval_nontriviali(
// CHECK: icmp ule i64 {{.*}}, 4
void x_byval_nontrivial(int i) { ntsink(nta[i]); }

//===----------------------------------------------------------------------===//
// Address-only contexts: `ule` is required.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}}@_Z9xctl_addri(
// CHECK: icmp ule i64 {{.*}}, 4
void xctl_addr(int i) { p = &a[i]; }

// CHECK-LABEL: define {{.*}}@_Z8xctl_rowi(
// CHECK: icmp ule i64 {{.*}}, 4
void xctl_row(int i) { p = a2[i]; }

// Same expression as xctl_row.
// CHECK-LABEL: define {{.*}}@_Z14xctl_row_elem0i(
// CHECK: icmp ule i64 {{.*}}, 4
void xctl_row_elem0(int i) { p = &a2[i][0]; }

// CHECK-LABEL: define {{.*}}@_Z16xctl_struct_addri(
// CHECK: icmp ule i64 {{.*}}, 4
void xctl_struct_addr(int i) { q = &ma[i]; }

// The pointer form of the conversion in x_derived_to_base, which
// is arithmetic rather than a subobject designation.
// CHECK-LABEL: define {{.*}}@_Z13xctl_base_ptri(
// CHECK: icmp ule i64 {{.*}}, 4
Base *xctl_base_ptr(int i) { return &da[i]; }

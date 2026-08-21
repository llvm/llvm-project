// Baseline record of the comparison -fsanitize=array-bounds emits for each
// context a subscript can appear in. `ult` requires the element to exist; `ule`
// also accepts an index equal to the bound, which is right only where the
// expression forms an address without reaching the object (C99 6.5.6p8,
// C99 6.5.3.2p3).
//
// A case named `..._arrow` spells the same access as its non-arrow counterpart,
// through a pointer instead; C99 6.5.2.3p4 makes them one expression, so the two
// must agree.
//
// Later commits in this series change individual CHECK lines, so each shows in
// its diff exactly which contexts it affects.
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=array-bounds \
// RUN:     -Wno-array-bounds -std=c11 %s -o - | FileCheck %s
//
// The __block section needs -fblocks, so it is checked under its own prefix.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=array-bounds \
// RUN:     -Wno-array-bounds -std=c11 -fblocks -DBLOCKS %s -o - \
// RUN:     | FileCheck %s --check-prefixes=CHECK,BLOCKS

struct S {
  int x;
};
typedef int v4 __attribute__((ext_vector_type(4)));
struct CB {
  int n;
  int fam[] __attribute__((counted_by(n)));
};

int a[4];
int a2[4][4];
struct S sa[4];
_Complex double ca[4];
v4 va[4];
int *p;
struct S *q;
int v;
struct S aggl;
_Complex double cv;
void sink(struct S);

//===----------------------------------------------------------------------===//
// Contexts that require the element to exist.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}}@c_store(
// CHECK: icmp ult i64 {{.*}}, 4
void c_store(int i) { a[i] = 1; }

// CHECK-LABEL: define {{.*}}@c_load(
// CHECK: icmp ult i64 {{.*}}, 4
void c_load(int i) { v = a[i]; }

// CHECK-LABEL: define {{.*}}@c_load_deref_addr(
// CHECK: icmp ule i64 {{.*}}, 4
void c_load_deref_addr(int i) { v = *&a[i]; }

// CHECK-LABEL: define {{.*}}@c_load_cast(
// CHECK: icmp ule i64 {{.*}}, 4
void c_load_cast(int i) { v = *(int *)&a[i]; }

// CHECK-LABEL: define {{.*}}@c_compound(
// CHECK: icmp ult i64 {{.*}}, 4
void c_compound(int i) { a[i] += 1; }

// CHECK-LABEL: define {{.*}}@c_compound_paren(
// CHECK: icmp ult i64 {{.*}}, 4
void c_compound_paren(int i) { (a[i]) += 1; }

// CHECK-LABEL: define {{.*}}@c_compound_deref_addr(
// CHECK: icmp ult i64 {{.*}}, 4
void c_compound_deref_addr(int i) { *&a[i] += 1; }

// CHECK-LABEL: define {{.*}}@c_postinc(
// CHECK: icmp ult i64 {{.*}}, 4
void c_postinc(int i) { a[i]++; }

// CHECK-LABEL: define {{.*}}@c_postdec(
// CHECK: icmp ult i64 {{.*}}, 4
void c_postdec(int i) { a[i]--; }

// CHECK-LABEL: define {{.*}}@c_preinc(
// CHECK: icmp ult i64 {{.*}}, 4
void c_preinc(int i) { ++a[i]; }

// CHECK-LABEL: define {{.*}}@c_predec(
// CHECK: icmp ult i64 {{.*}}, 4
void c_predec(int i) { --a[i]; }

// CHECK-LABEL: define {{.*}}@c_agg_store(
// CHECK: icmp ult i64 {{.*}}, 4
void c_agg_store(int i) { sa[i] = aggl; }

// CHECK-LABEL: define {{.*}}@c_agg_store_paren(
// CHECK: icmp ult i64 {{.*}}, 4
void c_agg_store_paren(int i) { (sa[i]) = aggl; }

// CHECK-LABEL: define {{.*}}@c_agg_store_deref_addr(
// CHECK: icmp ult i64 {{.*}}, 4
void c_agg_store_deref_addr(int i) { *&sa[i] = aggl; }

// CHECK-LABEL: define {{.*}}@c_agg_load(
// CHECK: icmp ult i64 {{.*}}, 4
void c_agg_load(int i) { aggl = sa[i]; }

// CHECK-LABEL: define {{.*}}@c_agg_load_deref_addr(
// CHECK: icmp ule i64 {{.*}}, 4
void c_agg_load_deref_addr(int i) { aggl = *&sa[i]; }

// CHECK-LABEL: define {{.*}}@c_member_dot(
// CHECK: icmp ult i64 {{.*}}, 4
void c_member_dot(int i) { sa[i].x = 1; }

// CHECK-LABEL: define {{.*}}@c_member_dot_deref_addr(
// CHECK: icmp ult i64 {{.*}}, 4
void c_member_dot_deref_addr(int i) { (*&sa[i]).x = 1; }

// CHECK-LABEL: define {{.*}}@c_member_dot_deref_addr_load(
// CHECK: icmp ult i64 {{.*}}, 4
void c_member_dot_deref_addr_load(int i) { v = (*&sa[i]).x; }

// CHECK-LABEL: define {{.*}}@c_member_arrow(
// CHECK: icmp ult i64 {{.*}}, 4
void c_member_arrow(int i) { (&sa[i])->x = 1; }

// Taking the address of a member is not one of the rewrites in C99 6.5.3.2p3,
// so unlike ctl_struct_addr these require the element to exist.
// CHECK-LABEL: define {{.*}}@c_member_dot_addr(
// CHECK: icmp ult i64 {{.*}}, 4
void c_member_dot_addr(int i) { p = &sa[i].x; }

// CHECK-LABEL: define {{.*}}@c_member_arrow_addr(
// CHECK: icmp ult i64 {{.*}}, 4
void c_member_arrow_addr(int i) { p = &(&sa[i])->x; }

// CHECK-LABEL: define {{.*}}@c_complex_real(
// CHECK: icmp ult i64 {{.*}}, 4
void c_complex_real(int i) { __real__ ca[i] = 1; }

// CHECK-LABEL: define {{.*}}@c_complex_imag(
// CHECK: icmp ult i64 {{.*}}, 4
void c_complex_imag(int i) { __imag__ ca[i] = 1; }

// CHECK-LABEL: define {{.*}}@c_complex_load(
// CHECK: icmp ult i64 {{.*}}, 4
void c_complex_load(int i) { cv = ca[i]; }

// CHECK-LABEL: define {{.*}}@c_complex_store(
// CHECK: icmp ult i64 {{.*}}, 4
void c_complex_store(int i) { ca[i] = cv; }

// CHECK-LABEL: define {{.*}}@c_complex_compound(
// CHECK: icmp ult i64 {{.*}}, 4
void c_complex_compound(int i) { ca[i] += cv; }

// CHECK-LABEL: define {{.*}}@c_complex_incdec(
// CHECK: icmp ult i64 {{.*}}, 4
void c_complex_incdec(int i) { ca[i]++; }

// CHECK-LABEL: define {{.*}}@c_vec_elem(
// CHECK: icmp ult i64 {{.*}}, 4
void c_vec_elem(int i) { va[i].x = 1; }

// CHECK-LABEL: define {{.*}}@c_vec_elem_arrow(
// CHECK: icmp ult i64 {{.*}}, 4
void c_vec_elem_arrow(int i) { (&va[i])->x = 1; }

// The component belongs to a temporary, not to an element; the
// element is read in order to build it.
// CHECK-LABEL: define {{.*}}@c_vec_rvalue(
// CHECK: icmp ult i64 {{.*}}, 4
void c_vec_rvalue(int i) {
  v = (va[i] + va[0]).x;
}

// CHECK-LABEL: define {{.*}}@c_byval(
// CHECK: icmp ult i64 {{.*}}, 4
void c_byval(int i) { sink(sa[i]); }

// Same expression as c_store: C99 6.5.3.2p3 makes `&*E` into `E`.
// CHECK-LABEL: define {{.*}}@c_deref_addr(
// CHECK: icmp ult i64 {{.*}}, 4
void c_deref_addr(int i) { *&a[i] = 1; }

// CHECK-LABEL: define {{.*}}@c_paren(
// CHECK: icmp ult i64 {{.*}}, 4
void c_paren(int i) { (a[i]) = 1; }

// CHECK-LABEL: define {{.*}}@c_extension(
// CHECK: icmp ult i64 {{.*}}, 4
void c_extension(int i) { __extension__(a[i]) = 1; }

// CHECK-LABEL: define {{.*}}@c_cast_store(
// CHECK: icmp ult i64 {{.*}}, 4
void c_cast_store(int i) { *(int *)&a[i] = 1; }

// CHECK-LABEL: define {{.*}}@c_counted(
// CHECK: icmp ult i32 {{.*}}
void c_counted(struct CB *s, int i) { s->fam[i] = 1; }

//===----------------------------------------------------------------------===//
// __block storage. AggExprEmitter::VisitBinAssign has a separate path for a
// __block LHS whose RHS has side effects -- the comment there calls it
// "pretty semantically fragile" -- so both shapes are covered: only the first
// reaches that path.
//===----------------------------------------------------------------------===//

#ifdef BLOCKS
struct S mk(void);

// A side-effecting right operand takes a different path from
// blk_plain below.
// BLOCKS-LABEL: define {{.*}}@blk_side_effect(
// BLOCKS: icmp ult i64 {{.*}}, 4
void blk_side_effect(int i) {
  __block struct S barr[4];
  barr[i] = mk();
  (void)barr;
}

// BLOCKS-LABEL: define {{.*}}@blk_plain(
// BLOCKS: icmp ult i64 {{.*}}, 4
void blk_plain(int i, struct S v) {
  __block struct S barr[4];
  barr[i] = v;
  (void)barr;
}
#endif

//===----------------------------------------------------------------------===//
// Address-only contexts. `ule` is required here: these expressions form an
// address without reaching the object, so a strict comparison would reject
// correct code.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}}@ctl_addr(
// CHECK: icmp ule i64 {{.*}}, 4
void ctl_addr(int i) { p = &a[i]; }

// CHECK-LABEL: define {{.*}}@ctl_decay(
// CHECK: icmp ule i64 {{.*}}, 4
void ctl_decay(int i) { p = a + i; }

// CHECK-LABEL: define {{.*}}@ctl_row(
// CHECK: icmp ule i64 {{.*}}, 4
void ctl_row(int i) { p = a2[i]; }

// Same expression as ctl_row: C99 6.5.3.2p3.
// CHECK-LABEL: define {{.*}}@ctl_row_elem0(
// CHECK: icmp ule i64 {{.*}}, 4
void ctl_row_elem0(int i) { p = &a2[i][0]; }

// CHECK-LABEL: define {{.*}}@ctl_struct_addr(
// CHECK: icmp ule i64 {{.*}}, 4
void ctl_struct_addr(int i) { q = &sa[i]; }

// CHECK-LABEL: define {{.*}}@ctl_addr_deref(
// CHECK: icmp ule i64 {{.*}}, 4
void ctl_addr_deref(int i) { p = &*&a[i]; }

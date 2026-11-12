// RUN: %clang_cc1 -triple=x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LLVM
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -DCIR -triple=x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s %}

struct s {
  char filler [128];
  volatile int x;
};

struct s gs;

void foo (void) {
  struct s ls;
  ls = ls;
  gs = gs;
  ls = gs;
}
// CHECK-LABEL: define{{.*}} void @foo()
// CHECK: %[[LS:.*]] = alloca %struct.s, align 4
// CHECK-NEXT: call void @llvm.memcpy.{{.*}}(ptr align 4 %[[LS]], ptr align 4 %[[LS]], i64 132, i1 true)
// CHECK-NEXT: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs, ptr align 4 @gs, i64 132, i1 true)
// CHECK-NEXT: call void @llvm.memcpy.{{.*}}(ptr align 4 %[[LS]], ptr align 4 @gs, i64 132, i1 true)


struct s1 {
  struct s y;
};

struct s1 s;

void fee (void) {
  s = s;
  s.y = gs;
}
// CHECK-LABEL: define{{.*}} void @fee()
// CHECK: call void @llvm.memcpy.{{.*}}(ptr align 4 @s, ptr align 4 @s, i64 132, i1 true)
// CHECK-NEXT: call void @llvm.memcpy.{{.*}}(ptr align 4 @s, ptr align 4 @gs, i64 132, i1 true)

union u {
  int i;
  volatile int vi;
};

struct s2 {
  char filler[128];
  union u u;
};

struct s2 gs2;
union u gu;

void union_zero_init(void) {
  gs2 = (struct s2){0};
}
// LLVM-LABEL: define{{.*}} void @union_zero_init()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs2, ptr align 4 {{.*}}, i64 132, i1 false)

void union_nonvolatile_field(void) {
  gs2 = (struct s2){ .u.i = 1 };
}
// LLVM-LABEL: define{{.*}} void @union_nonvolatile_field()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs2, ptr align 4 {{.*}}, i64 132, i1 false)

void union_nonvolatile_field_from_volatile_load(void) {
  gs2 = (struct s2){ .u.i = gs.x };
}
// LLVM-LABEL: define{{.*}} void @union_nonvolatile_field_from_volatile_load()
// LLVM: load volatile i32, ptr getelementptr inbounds nuw (i8, ptr @gs, i64 128)
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs2, ptr align 4 {{.*}}, i64 132, i1 false)

void union_nested_compound_literal(void) {
  gs2 = (struct s2){ .u = (union u){ .i = 1 } };
}
// LLVM-LABEL: define{{.*}} void @union_nested_compound_literal()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs2, ptr align 4 {{.*}}, i64 132, i1 false)

// ClangIR does not yet implement volatile aggregate lvalue-to-rvalue casts.
#ifndef CIR
void union_volatile_compound_literal(void) {
  gs2 = (struct s2){ .u = ((volatile union u){0}) };
}
// LLVM-LABEL: define{{.*}} void @union_volatile_compound_literal()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs2, ptr align 4 {{.*}}, i64 132, i1 true)
#endif

void union_volatile_field(void) {
  gs2 = (struct s2){ .u.vi = 1 };
}
// LLVM-LABEL: define{{.*}} void @union_volatile_field()
// LLVM: store volatile i32 1
// LLVM-NEXT: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs2, ptr align 4 {{.*}}, i64 132, i1 true)

void union_copy_subobject(void) {
  gs2 = (struct s2){ .u = gu };
}
// LLVM-LABEL: define{{.*}} void @union_copy_subobject()
// Note: Copying @gu into the temporary is non-volatile because aggregate
// initialization does not check volatility (see FIXME in EmitInitializationToLValue).
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 {{.*}}, ptr align 4 @gu, i64 4, i1 false)
// LLVM-NEXT: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs2, ptr align 4 {{.*}}, i64 132, i1 true)

void union_struct_assign(void) {
  gs2 = gs2;
}
// LLVM-LABEL: define{{.*}} void @union_struct_assign()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gs2, ptr align 4 @gs2, i64 132, i1 true)

union u_vfirst {
  int : 4;
  volatile int vi;
  int i;
};

union u_vfirst guvf;

void union_empty_init_nonvolatile_first(void) {
  gu = (union u){};
}
// LLVM-LABEL: define{{.*}} void @union_empty_init_nonvolatile_first()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gu, ptr align 4 {{.*}}, i64 4, i1 false)

void union_empty_init_volatile_first(void) {
  guvf = (union u_vfirst){};
}
// LLVM-LABEL: define{{.*}} void @union_empty_init_volatile_first()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @guvf, ptr align 4 {{.*}}, i64 4, i1 true)

void union_override_volatile_first(void) {
  guvf = (union u_vfirst){ .i = 1 };
}
// LLVM-LABEL: define{{.*}} void @union_override_volatile_first()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @guvf, ptr align 4 {{.*}}, i64 4, i1 false)

struct s_complex {
  volatile int : 4;
  union u u_arr[2];
  union u_vfirst uvf_arr[2];
  union u_vfirst uvf;
  union u trailing_u;
  volatile int flex[];
};

struct s_complex gsc;

void complex_nonvolatile(void) {
  // Unnamed volatile bitfield, omitted non-volatile union array (`!ILE`),
  // explicit non-volatile union array/field, omitted `trailing_u`, and
  // trailing volatile flexible array member -> non-volatile memcpy.
  gsc = (struct s_complex){ .uvf_arr = { { .i = 1 }, { .i = 2 } }, .uvf.i = 3 };
}
// LLVM-LABEL: define{{.*}} void @complex_nonvolatile()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gsc, ptr align 4 {{.*}}, i64 28, i1 false)

void complex_omitted_volatile_array(void) {
  gsc = (struct s_complex){ .uvf.i = 1 };
}
// LLVM-LABEL: define{{.*}} void @complex_omitted_volatile_array()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gsc, ptr align 4 {{.*}}, i64 28, i1 true)

void complex_explicit_volatile_array_elt(void) {
  gsc = (struct s_complex){ .u_arr = { { .i = 1 }, { .vi = 2 } },
                            .uvf_arr = { { .i = 3 }, { .i = 4 } },
                            .uvf.i = 5 };
}
// LLVM-LABEL: define{{.*}} void @complex_explicit_volatile_array_elt()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gsc, ptr align 4 {{.*}}, i64 28, i1 true)

void complex_nonvolatile_array_filler(void) {
  gsc = (struct s_complex){ .u_arr = { { .i = 1 } },
                            .uvf_arr = { { .i = 2 }, { .i = 3 } },
                            .uvf.i = 4 };
}
// LLVM-LABEL: define{{.*}} void @complex_nonvolatile_array_filler()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gsc, ptr align 4 {{.*}}, i64 28, i1 false)

void complex_volatile_array_filler(void) {
  gsc = (struct s_complex){ .uvf_arr = { { .i = 1 } }, .uvf.i = 2 };
}
// LLVM-LABEL: define{{.*}} void @complex_volatile_array_filler()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gsc, ptr align 4 {{.*}}, i64 28, i1 true)

void complex_omitted_trailing_volatile_field(void) {
  gsc = (struct s_complex){ .uvf_arr = { { .i = 1 }, { .i = 2 } } };
}
// LLVM-LABEL: define{{.*}} void @complex_omitted_trailing_volatile_field()
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 4 @gsc, ptr align 4 {{.*}}, i64 28, i1 true)

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-NOAVX --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclangir -emit-cir %s -o %t-avx.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-AVX --input-file=%t-avx.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR,NOAVX,NOAVX512 --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG,NOAVX,NOAVX512 --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -fclangir -emit-llvm %s -o %t-cir-avx.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR,AVX,NOAVX512 --input-file=%t-cir-avx.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm %s -o %t-avx.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG,AVX,NOAVX512 --input-file=%t-avx.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -fclangir -emit-llvm %s -o %t-cir-avx512.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR,AVX,AVX512 --input-file=%t-cir-avx512.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -emit-llvm %s -o %t-avx512.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG,AVX,AVX512 --input-file=%t-avx512.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -menable-no-nans -menable-no-infs -fclangir -emit-llvm %s -o %t-cir-finite.ll
// RUN: FileCheck --check-prefix=FINITE --input-file=%t-cir-finite.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -menable-no-nans -menable-no-infs -emit-llvm %s -o %t-finite.ll
// RUN: FileCheck --check-prefix=FINITE --input-file=%t-finite.ll %s

typedef char c3 __attribute__((ext_vector_type(3)));
typedef char c4 __attribute__((ext_vector_type(4)));
typedef short s3 __attribute__((ext_vector_type(3)));
typedef float f3 __attribute__((ext_vector_type(3)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef double d3 __attribute__((ext_vector_type(3)));
typedef float f5 __attribute__((ext_vector_type(5)));
typedef double d5 __attribute__((ext_vector_type(5)));
typedef int i6 __attribute__((ext_vector_type(6)));
typedef short s7 __attribute__((ext_vector_type(7)));

// A three-float vector takes 16 bytes, so the member after it is at 16.
struct S { f3 v; float x; };
float member_after_vec3(struct S *s) { return s->x; }

// CIR-DAG: !rec_S = !cir.struct<"S" {data !cir.vector<3 x !cir.float>, data !cir.float}>
// LLVM-DAG: %struct.S = type { <3 x float>, float }

// Each element of an array of them takes 16 bytes too.
struct T { f3 v[4]; int x; };
int member_after_vec3_array(struct T *t) { return t->x; }

// CIR-DAG: !rec_T = !cir.struct<"T" {data !cir.array<!cir.vector<3 x !cir.float> x 4>, data !s32i}>
// LLVM-DAG: %struct.T = type { [4 x <3 x float>], i32 }

struct N { struct S s; float y; };
float member_after_nested(struct N *n) { return n->y; }

// CIR-DAG: !rec_N = !cir.struct<"N" {data !rec_S, data !cir.float}>
// LLVM-DAG: %struct.N = type { %struct.S, float }

// A union holding one is 16 bytes, and between two variants of the same
// alignment and alloc size the first is the storage.
union U { f3 v; int i; };
union UF { f3 a; f4 b; };
union UF uf;

// CIR-DAG: !rec_U = !cir.union<"U" {data !cir.vector<3 x !cir.float>, data !s32i}>
// CIR-DAG: !rec_UF = !cir.union<"UF" {data !cir.vector<3 x !cir.float>, data !cir.vector<4 x !cir.float>}>
// LLVM-DAG: %union.U = type { <3 x float> }
// LLVM-DAG: %union.UF = type { <3 x float> }

// CIR-LABEL: cir.func{{.*}} @member_after_vec3(
// CIR: cir.get_member %{{.+}}[1] {name = "x"} : !cir.ptr<!rec_S> -> !cir.ptr<!cir.float>
// CIR-LABEL: cir.func{{.*}} @member_after_vec3_array(
// CIR: cir.get_member %{{.+}}[1] {name = "x"} : !cir.ptr<!rec_T> -> !cir.ptr<!s32i>
// CIR-LABEL: cir.func{{.*}} @member_after_nested(
// CIR: cir.get_member %{{.+}}[1] {name = "y"} : !cir.ptr<!rec_N> -> !cir.ptr<!cir.float>
// LLVM-LABEL: define dso_local float @member_after_vec3(
// LLVM: getelementptr inbounds nuw %struct.S, ptr %{{.+}}, i32 0, i32 1
// LLVM-NEXT: load float, ptr %{{.+}}, align 16
// LLVM-LABEL: define dso_local i32 @member_after_vec3_array(
// LLVM: getelementptr inbounds nuw %struct.T, ptr %{{.+}}, i32 0, i32 1
// LLVM-NEXT: load i32, ptr %{{.+}}, align 16
// LLVM-LABEL: define dso_local float @member_after_nested(
// LLVM: getelementptr inbounds nuw %struct.N, ptr %{{.+}}, i32 0, i32 1
// LLVM-NEXT: load float, ptr %{{.+}}, align 16

void copy_S(struct S *a, struct S *b) { *a = *b; }
void copy_T(struct T *a, struct T *b) { *a = *b; }
void copy_N(struct N *a, struct N *b) { *a = *b; }
void copy_U(union U *a, union U *b) { *a = *b; }

// CIR-LABEL: cir.func{{.*}} @copy_S(
// CIR: cir.copy %{{.+}} align(16) to %{{.+}} align(16) : !cir.ptr<!rec_S>
// CIR-LABEL: cir.func{{.*}} @copy_T(
// CIR: cir.copy %{{.+}} align(16) to %{{.+}} align(16) : !cir.ptr<!rec_T>
// CIR-LABEL: cir.func{{.*}} @copy_N(
// CIR: cir.copy %{{.+}} align(16) to %{{.+}} align(16) : !cir.ptr<!rec_N>
// CIR-LABEL: cir.func{{.*}} @copy_U(
// CIR: cir.copy %{{.+}} align(16) to %{{.+}} align(16) : !cir.ptr<!rec_U>
// LLVM-LABEL: define dso_local void @copy_S(
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.+}}, ptr align 16 %{{.+}}, i64 32, i1 false)
// LLVM-LABEL: define dso_local void @copy_T(
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.+}}, ptr align 16 %{{.+}}, i64 80, i1 false)
// LLVM-LABEL: define dso_local void @copy_N(
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.+}}, ptr align 16 %{{.+}}, i64 48, i1 false)
// LLVM-LABEL: define dso_local void @copy_U(
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.+}}, ptr align 16 %{{.+}}, i64 16, i1 false)

void byval_S(struct S s) {}
void byval_U(union U u) {}

// CIR: cir.func{{.*}} @byval_S(%arg0: !cir.ptr<!rec_S> {llvm.align = 16 : i64, llvm.byval = !rec_S, llvm.noundef}
// CIR: cir.func{{.*}} @byval_U(%arg0: !u64i loc({{.+}}), %arg1: !cir.double loc(
// LLVM-LABEL: define dso_local void @byval_S(ptr noundef byval(%struct.S) align 16 %{{.+}})
// LLVM-LABEL: define dso_local void @byval_U(i64 %{{.+}}, double %{{.+}})

// Three chars take 4 bytes, and the i32 they pass as has a byte that may be
// undef.  Four chars fill it.
void take_c3(c3 v) {}
c3 ret_c3(void) { return 0; }
void take_c4(c4 v) {}

// CIR: cir.func{{.*}} @take_c3(%arg0: !u32i loc(
// CIR: cir.func{{.*}} @ret_c3() -> !u32i
// CIR: cir.func{{.*}} @take_c4(%arg0: !u32i {llvm.noundef} loc(
// LLVM-LABEL: define dso_local void @take_c3(i32 %{{.+}})
// LLVM-LABEL: define dso_local i32 @ret_c3()
// LLVM-LABEL: define dso_local void @take_c4(i32 noundef %{{.+}})

void call_c3(void) { take_c3(ret_c3()); }
void var(int n, ...);
void call_var_c3(void) { var(1, ret_c3()); }
struct S ret_S_take_c3(c3 v);
void call_sret(void) { ret_S_take_c3(ret_c3()); }

// CIR: cir.call @take_c3(%{{.+}}) : (!u32i) -> ()
// CIR: cir.call @var(%{{.+}}, %{{.+}}) : (!s32i {llvm.noundef}, !u32i) -> ()
// CIR: cir.call @ret_S_take_c3(%{{.+}}, %{{.+}}) : (!cir.ptr<!rec_S> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.sret = !rec_S, llvm.writable}, !u32i) -> ()
// LLVM-LABEL: define dso_local void @call_c3()
// LLVM: call i32 @ret_c3()
// LLVM: call void @take_c3(i32 %{{.+}})
// LLVM-LABEL: define dso_local void @call_var_c3()
// LLVM: call i32 @ret_c3()
// LLVM: call void (i32, ...) @var(i32 noundef 1, i32 %{{.+}})
// LLVM-LABEL: define dso_local void @call_sret()
// LLVM: call i32 @ret_c3()
// LLVM: call void @ret_S_take_c3(ptr dead_on_unwind writable sret(%struct.S) align 16 %{{.+}}, i32 %{{.+}})

// Three shorts take 8 bytes.
void take_s3(s3 v) {}

// CIR: cir.func{{.*}} @take_s3(%arg0: !cir.double loc(
// LLVM-LABEL: define dso_local void @take_s3(double %{{.+}})

// Three floats take 16 bytes.
void take_f3(f3 v) {}

// CIR: cir.func{{.*}} @take_f3(%arg0: !cir.vector<3 x !cir.float> {llvm.noundef} loc(
// LLVM-LABEL: define dso_local void @take_f3(<3 x float> noundef %{{.+}})
// FINITE-LABEL: define dso_local void @take_f3(<3 x float> noundef nofpclass(nan inf) %{{.+}})

struct AF3 { f3 v[1]; };
struct AF3 pass_af3(struct AF3 s) { return s; }
struct A2F3 { f3 v[2]; };
void take_a2f3(struct A2F3 s) {}

void take_uf(union UF u) {}
union UFF { f3 v; float f[3]; };
void take_uff(union UFF u) {}
struct SS3S { s3 v; short x; };
void take_ss3s(struct SS3S s) {}
typedef _Float16 h3 __attribute__((ext_vector_type(3)));
void take_h3(h3 v) {}

// CIR: cir.func{{.*}} @pass_af3(%arg0: !cir.vector<3 x !cir.float> loc({{.+}})) -> !cir.vector<3 x !cir.float>
// CIR: cir.func{{.*}} @take_a2f3(%arg0: !cir.ptr<!rec_A2F3> {llvm.align = 16 : i64, llvm.byval = !rec_A2F3, llvm.noundef} loc(
// CIR: cir.func{{.*}} @take_uf(%arg0: !cir.vector<2 x !cir.double> loc(
// CIR: cir.func{{.*}} @take_uff(%arg0: !cir.double loc({{.+}}), %arg1: !cir.double loc(
// CIR: cir.func{{.*}} @take_ss3s(%arg0: !cir.double loc({{.+}}), %arg1: !s16i loc(
// CIR: cir.func{{.*}} @take_h3(%arg0: !cir.double loc(
// LLVM-LABEL: define dso_local <3 x float> @pass_af3(<3 x float> %{{.+}})
// LLVM-LABEL: define dso_local void @take_a2f3(ptr noundef byval(%struct.A2F3) align 16 %{{.+}})
// LLVM-LABEL: define dso_local void @take_uf(<2 x double> %{{.+}})
// LLVM-LABEL: define dso_local void @take_uff(double %{{.+}}, double %{{.+}})
// LLVM-LABEL: define dso_local void @take_ss3s(double %{{.+}}, i16 %{{.+}})
// LLVM-LABEL: define dso_local void @take_h3(double %{{.+}})
// FINITE-LABEL: define dso_local void @take_h3(double nofpclass(nan inf) %{{.+}})

// Each half of a flattened argument carries the argument's attributes.
void take_cd(_Complex double c) {}

// CIR: cir.func{{.*}} @take_cd(%arg0: !cir.double {llvm.noundef} loc({{.+}}), %arg1: !cir.double {llvm.noundef} loc(
// LLVM-LABEL: define dso_local void @take_cd(double noundef %{{.+}}, double noundef %{{.+}})
// FINITE-LABEL: define dso_local void @take_cd(double noundef nofpclass(nan inf) %{{.+}}, double noundef nofpclass(nan inf) %{{.+}})

// Seven shorts take 16 bytes.
s7 pass_s7(s7 v) { return v; }

// CIR: cir.func{{.*}} @pass_s7(%arg0: !cir.vector<7 x !s16i> {llvm.noundef} loc({{.+}})) -> !cir.vector<7 x !s16i>
// LLVM-LABEL: define dso_local <7 x i16> @pass_s7(<7 x i16> noundef %{{.+}})

// Three doubles, five floats and six ints take 32 bytes, which AVX passes
// directly.
void take_d3(d3 v) {}
f5 pass_f5(f5 v) { return v; }
i6 pass_i6(i6 v) { return v; }

// CIR-NOAVX: cir.func{{.*}} @take_d3(%arg0: !cir.ptr<!cir.vector<3 x !cir.double>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<3 x !cir.double>, llvm.noundef}
// CIR-NOAVX: cir.func{{.*}} @pass_f5(%arg0: !cir.ptr<!cir.vector<5 x !cir.float>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<5 x !cir.float>, llvm.noundef} loc({{.+}})) -> !cir.vector<5 x !cir.float>
// CIR-NOAVX: cir.func{{.*}} @pass_i6(%arg0: !cir.ptr<!cir.vector<6 x !s32i>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<6 x !s32i>, llvm.noundef} loc({{.+}})) -> !cir.vector<6 x !s32i>
// CIR-AVX: cir.func{{.*}} @take_d3(%arg0: !cir.vector<3 x !cir.double> {llvm.noundef} loc(
// CIR-AVX: cir.func{{.*}} @pass_f5(%arg0: !cir.vector<5 x !cir.float> {llvm.noundef} loc({{.+}})) -> !cir.vector<5 x !cir.float>
// CIR-AVX: cir.func{{.*}} @pass_i6(%arg0: !cir.vector<6 x !s32i> {llvm.noundef} loc({{.+}})) -> !cir.vector<6 x !s32i>
// NOAVX-LABEL: define dso_local void @take_d3(ptr noundef byval(<3 x double>) align 32 %{{.+}})
// NOAVX-LABEL: define dso_local <5 x float> @pass_f5(ptr noundef byval(<5 x float>) align 32 %{{.+}})
// NOAVX-LABEL: define dso_local <6 x i32> @pass_i6(ptr noundef byval(<6 x i32>) align 32 %{{.+}})
// AVX-LABEL: define dso_local void @take_d3(<3 x double> noundef %{{.+}})
// AVX-LABEL: define dso_local <5 x float> @pass_f5(<5 x float> noundef %{{.+}})
// AVX-LABEL: define dso_local <6 x i32> @pass_i6(<6 x i32> noundef %{{.+}})
// A byval pointer does not carry the vector's nofpclass.
// FINITE-LABEL: define dso_local void @take_d3(ptr noundef byval(<3 x double>) align 32 %{{.+}})
// FINITE-LABEL: define dso_local nofpclass(nan inf) <5 x float> @pass_f5(ptr noundef byval(<5 x float>) align 32 %{{.+}})

struct SF5 { f5 v; };
struct SF5 pass_sf5(struct SF5 s) { return s; }

// CIR-NOAVX: cir.func{{.*}} @pass_sf5(%arg0: !cir.ptr<!rec_SF5> {llvm.align = 32 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_SF5, llvm.writable} loc({{.+}}), %arg1: !cir.ptr<!rec_SF5> {llvm.align = 32 : i64, llvm.byval = !rec_SF5, llvm.noundef}
// CIR-AVX: cir.func{{.*}} @pass_sf5(%arg0: !cir.vector<5 x !cir.float> loc({{.+}})) -> !cir.vector<5 x !cir.float>
// NOAVX-LABEL: define dso_local void @pass_sf5(ptr dead_on_unwind noalias writable sret(%struct.SF5) align 32 %{{.+}}, ptr noundef byval(%struct.SF5) align 32 %{{.+}})
// AVX-LABEL: define dso_local <5 x float> @pass_sf5(<5 x float> %{{.+}})

f5 src_f5(void);
void take_f5(f5 v);
void call_f5(void) { take_f5(src_f5()); }

// CIR-NOAVX: cir.call @take_f5(%{{.+}}) : (!cir.ptr<!cir.vector<5 x !cir.float>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<5 x !cir.float>, llvm.noundef}) -> ()
// CIR-AVX: cir.call @take_f5(%{{.+}}) : (!cir.vector<5 x !cir.float> {llvm.noundef}) -> ()
// LLVM-LABEL: define dso_local void @call_f5()
// NOAVX: call void @take_f5(ptr noundef byval(<5 x float>) align 32 %{{.+}})
// AVX: call void @take_f5(<5 x float> noundef %{{.+}})
// FINITE-LABEL: define dso_local void @call_f5()
// FINITE: call void @take_f5(ptr noundef byval(<5 x float>) align 32 %{{.+}})

// Five doubles take 64 bytes, which only AVX-512 passes directly.
void take_d5(d5 v) {}

// CIR: cir.func{{.*}} @take_d5(%arg0: !cir.ptr<!cir.vector<5 x !cir.double>> {llvm.align = 64 : i64, llvm.byval = !cir.vector<5 x !cir.double>, llvm.noundef} loc(
// NOAVX512-LABEL: define dso_local void @take_d5(ptr noundef byval(<5 x double>) align 64 %{{.+}})
// AVX512-LABEL: define dso_local void @take_d5(<5 x double> noundef %{{.+}})

// On the stack a three-char vector is coerced to i32 too.
void stack_c3(long a, long b, long c, long d, long e, long f, c3 v) {}

// CIR: cir.func{{.*}} @stack_c3(%arg0: !s64i {llvm.noundef} loc({{.+}}), %arg1: !s64i {llvm.noundef} loc({{.+}}), %arg2: !s64i {llvm.noundef} loc({{.+}}), %arg3: !s64i {llvm.noundef} loc({{.+}}), %arg4: !s64i {llvm.noundef} loc({{.+}}), %arg5: !s64i {llvm.noundef} loc({{.+}}), %arg6: !u32i loc(
// LLVM-LABEL: define dso_local void @stack_c3(i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i32 %{{.+}})

// A vector read from the overflow area is followed by the next argument 32
// bytes on.
f5 va_f5(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  f5 v = __builtin_va_arg(ap, f5);
  __builtin_va_end(ap);
  return v;
}

// CIR-LABEL: cir.func{{.*}} @va_f5(
// CIR: %[[STRIDE:[0-9]+]] = cir.const #cir.int<32> : !s32i
// CIR-NEXT: cir.ptr_stride %{{[0-9]+}}, %[[STRIDE]] : (!cir.ptr<!u8i>, !s32i) -> !cir.ptr<!u8i>
// LLVM-LABEL: define dso_local <5 x float> @va_f5(
// LLVM: call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -32)
// LLVMCIR: getelementptr i8, ptr %{{.+}}, i64 32
// OGCG: getelementptr i8, ptr %{{.+}}, i32 32

s7 va_s7(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  s7 v = __builtin_va_arg(ap, s7);
  __builtin_va_end(ap);
  return v;
}

// Seven shorts come from an xmm slot, or the overflow area in 16-byte steps.
// LLVM-LABEL: define dso_local <7 x i16> @va_s7(
// LLVM: icmp ule i32 %{{.+}}, 160
// LLVM: add i32 %{{.+}}, 16
// LLVM: call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -16)
// LLVMCIR: getelementptr i8, ptr %{{.+}}, i64 16
// OGCG: getelementptr i8, ptr %{{.+}}, i32 16
// LLVM: load <7 x i16>, ptr %{{.+}}, align 16

// Pointer subtraction divides by the 16- and 4-byte alloc sizes.
long diff_f3(f3 *a, f3 *b) { return a - b; }
long diff_c3(c3 *a, c3 *b) { return a - b; }

// CIR: cir.ptr_diff %{{.+}}, %{{.+}} : !cir.ptr<!cir.vector<3 x !cir.float>> -> !s64i
// CIR: cir.ptr_diff %{{.+}}, %{{.+}} : !cir.ptr<!cir.vector<3 x !s8i>> -> !s64i
// LLVM-LABEL: define dso_local i64 @diff_f3(
// LLVM: sdiv exact i64 %{{.+}}, 16
// LLVM-LABEL: define dso_local i64 @diff_c3(
// LLVM: sdiv exact i64 %{{.+}}, 4

void call_c4(c4 v) { take_c4(v); }

// CIR: cir.call @take_c4(%{{.+}}) : (!u32i {llvm.noundef}) -> ()
// LLVM-LABEL: define dso_local void @call_c4(
// LLVM: call void @take_c4(i32 noundef %{{.+}})

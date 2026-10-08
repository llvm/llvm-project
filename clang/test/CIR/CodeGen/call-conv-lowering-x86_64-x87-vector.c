// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-SSE --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-SSE --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-SSE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -target-feature +avx -fclangir -emit-cir %s -o %t-avx.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-AVX --input-file=%t-avx.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -target-feature +avx -fclangir -emit-llvm %s -o %t-avx-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX --input-file=%t-avx-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -target-feature +avx -emit-llvm %s -o %t-avx.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX --input-file=%t-avx.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -target-feature +avx512f -fclangir -emit-cir %s -o %t-avx512.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-AVX512 --input-file=%t-avx512.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -target-feature +avx512f -fclangir -emit-llvm %s -o %t-avx512-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX512 --input-file=%t-avx512-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-psabi -target-feature +avx512f -emit-llvm %s -o %t-avx512.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX512 --input-file=%t-avx512.ll %s

typedef long double ld1 __attribute__((vector_size(16)));
typedef long double ld2 __attribute__((vector_size(32)));
typedef long double ld4 __attribute__((vector_size(64)));

// A one-element x87 vector occupies 16 bytes and passes in a register as
// itself, so the value keeps its long double element and is not coerced.
ld1 take_ld1(ld1 v) { return v; }

// CIR: cir.func {{.*}}@take_ld1(%arg0: !cir.vector<1 x !cir.long_double<!cir.f80>> {llvm.noundef} loc({{[^)]+}})) -> !cir.vector<1 x !cir.long_double<!cir.f80>>
// CIR: cir.store %arg0, %{{.+}} : !cir.vector<1 x !cir.long_double<!cir.f80>>, !cir.ptr<!cir.vector<1 x !cir.long_double<!cir.f80>>>
// LLVM: define dso_local <1 x x86_fp80> @take_ld1(<1 x x86_fp80> noundef %{{[^,)]+}})

ld1 call_ld1(ld1 v) { return take_ld1(v); }

// CIR: cir.call @take_ld1(%{{.+}}) : (!cir.vector<1 x !cir.long_double<!cir.f80>> {llvm.noundef}) -> !cir.vector<1 x !cir.long_double<!cir.f80>>
// LLVM: call <1 x x86_fp80> @take_ld1(<1 x x86_fp80> noundef %{{[^,)]+}})

// A record holding only that vector, directly, in a one-element array, as a
// union member, or inside another record, is coerced to a vector of the
// underlying x87 format.
struct S1 { ld1 v; };
struct S1 take_s1(struct S1 s) { return s; }

// CIR: cir.func {{.*}}@take_s1(%arg0: !cir.vector<1 x !cir.f80> loc({{[^)]+}})) -> !cir.vector<1 x !cir.f80>
// LLVM: define dso_local <1 x x86_fp80> @take_s1(<1 x x86_fp80> %{{[^,)]+}})

struct S1 call_s1(struct S1 s) { return take_s1(s); }

// CIR: cir.call @take_s1(%{{.+}}) : (!cir.vector<1 x !cir.f80>) -> !cir.vector<1 x !cir.f80>
// LLVM: call <1 x x86_fp80> @take_s1(<1 x x86_fp80> %{{[^,)]+}})

struct A1 { ld1 a[1]; };
struct A1 take_a1(struct A1 a) { return a; }

// CIR: cir.func {{.*}}@take_a1(%arg0: !cir.vector<1 x !cir.f80> loc({{[^)]+}})) -> !cir.vector<1 x !cir.f80>
// LLVM: define dso_local <1 x x86_fp80> @take_a1(<1 x x86_fp80> %{{[^,)]+}})

union U1 { ld1 v; };
union U1 take_u1(union U1 u) { return u; }

// CIR: cir.func {{.*}}@take_u1(%arg0: !cir.vector<1 x !cir.f80> loc({{[^)]+}})) -> !cir.vector<1 x !cir.f80>
// LLVM: define dso_local <1 x x86_fp80> @take_u1(<1 x x86_fp80> %{{[^,)]+}})

struct N1 { struct S1 s; };
struct N1 take_n1(struct N1 n) { return n; }

// CIR: cir.func {{.*}}@take_n1(%arg0: !cir.vector<1 x !cir.f80> loc({{[^)]+}})) -> !cir.vector<1 x !cir.f80>
// LLVM: define dso_local <1 x x86_fp80> @take_n1(<1 x x86_fp80> %{{[^,)]+}})

// A struct holding anything past the one vector is larger than 16 bytes, so
// it passes in memory.
struct P2 { ld1 a; ld1 b; };
struct P2 take_p2(struct P2 p) { return p; }

// CIR: cir.func {{.*}}@take_p2(%arg0: !cir.ptr<!rec_P2> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_P2, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_P2> {llvm.align = 16 : i64, llvm.byval = !rec_P2, llvm.noundef} loc({{[^)]+}}))
// LLVM: define dso_local void @take_p2(ptr dead_on_unwind noalias writable sret(%struct.P2) align 16 %{{[^,)]+}}, ptr noundef byval(%struct.P2) align 16 %{{[^,)]+}})

struct M1 { int i; ld1 v; };
struct M1 take_m1(struct M1 m) { return m; }

// CIR: cir.func {{.*}}@take_m1(%arg0: !cir.ptr<!rec_M1> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_M1, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_M1> {llvm.align = 16 : i64, llvm.byval = !rec_M1, llvm.noundef} loc({{[^)]+}}))
// LLVM: define dso_local void @take_m1(ptr dead_on_unwind noalias writable sret(%struct.M1) align 16 %{{[^,)]+}}, ptr noundef byval(%struct.M1) align 16 %{{[^,)]+}})

// An unnamed argument up to 16 bytes is passed the same way as a named one.
void var(int n, ...);
void call_var(struct S1 s) { var(1, s); }

// CIR: cir.call @var(%{{.+}}, %{{.+}}) : (!s32i {llvm.noundef}, !cir.vector<1 x !cir.f80>) -> ()
// LLVM: call void (i32, ...) @var(i32 noundef 1, <1 x x86_fp80> %{{[^,)]+}})

void var_ld1(ld1 v) { var(1, v); }

// CIR: cir.call @var(%{{.+}}, %{{.+}}) : (!s32i {llvm.noundef}, !cir.vector<1 x !cir.long_double<!cir.f80>> {llvm.noundef}) -> ()
// LLVM: call void (i32, ...) @var(i32 noundef 1, <1 x x86_fp80> noundef %{{[^,)]+}})

// A wider vector takes the same IR form unnamed as named: it classifies as
// memory when unnamed, but a vector is still passed directly.
void var_ld2(ld2 v) { var(1, v); }

// CIR-SSE: cir.call @var(%{{.+}}, %{{.+}}) : (!s32i {llvm.noundef}, !cir.ptr<!cir.vector<2 x !cir.long_double<!cir.f80>>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<2 x !cir.long_double<!cir.f80>>, llvm.noundef}) -> ()
// CIR-AVX: cir.call @var(%{{.+}}, %{{.+}}) : (!s32i {llvm.noundef}, !cir.vector<2 x !cir.long_double<!cir.f80>> {llvm.noundef}) -> ()
// CIR-AVX512: cir.call @var(%{{.+}}, %{{.+}}) : (!s32i {llvm.noundef}, !cir.vector<2 x !cir.long_double<!cir.f80>> {llvm.noundef}) -> ()
// LLVM-SSE: call void (i32, ...) @var(i32 noundef 1, ptr noundef byval(<2 x x86_fp80>) align 32 %{{[^,)]+}})
// LLVM-AVX: call void (i32, ...) @var(i32 noundef 1, <2 x x86_fp80> noundef %{{[^,)]+}})
// LLVM-AVX512: call void (i32, ...) @var(i32 noundef 1, <2 x x86_fp80> noundef %{{[^,)]+}})

// va_arg takes a one-element vector from an SSE register slot, 16 bytes at a
// time, or from the overflow area aligned to 16.
ld1 arg_ld1(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  ld1 v = __builtin_va_arg(ap, ld1);
  __builtin_va_end(ap);
  return v;
}

// CIR-LABEL: cir.func {{.*}}@arg_ld1(
// CIR: %[[FP_OFFSET_P:.+]] = cir.get_member %{{.+}}[1] {name = "fp_offset"}
// CIR: %[[FP_OFFSET:.+]] = cir.load %[[FP_OFFSET_P]]
// CIR: %[[FP_LIMIT:.+]] = cir.const #cir.int<160> : !u32i
// CIR: cir.cmp le %[[FP_OFFSET]], %[[FP_LIMIT]] : !u32i
// CIR: %[[FP_STEP:.+]] = cir.const #cir.int<16> : !u32i
// CIR: cir.add %[[FP_OFFSET]], %[[FP_STEP]] : !u32i
// CIR: %[[ALIGN_MASK:.+]] = cir.const #cir.int<-16> : !s64i
// CIR: %[[ALIGNED:.+]] = cir.ptr_mask %{{.+}}, %[[ALIGN_MASK]]
// CIR: %[[MEM_STEP:.+]] = cir.const #cir.int<16> : !s32i
// CIR: cir.ptr_stride %[[ALIGNED]], %[[MEM_STEP]]
// CIR: %[[VPTR:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!u8i> -> !cir.ptr<!cir.vector<1 x !cir.long_double<!cir.f80>>>
// CIR: cir.load align(16) %[[VPTR]] : !cir.ptr<!cir.vector<1 x !cir.long_double<!cir.f80>>>, !cir.vector<1 x !cir.long_double<!cir.f80>>

// LLVM-LABEL: define dso_local <1 x x86_fp80> @arg_ld1(i32 noundef %{{[^,)]+}}, ...)
// LLVM: icmp ule i32 %{{.+}}, 160
// LLVM: add i32 %{{.+}}, 16
// LLVM: call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -16)
// LLVM: %[[ADDR:.+]] = phi ptr
// LLVM: load <1 x x86_fp80>, ptr %[[ADDR]], align 16

// A wider vector is always taken from the overflow area, aligned to its size.
ld2 arg_ld2(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  ld2 v = __builtin_va_arg(ap, ld2);
  __builtin_va_end(ap);
  return v;
}

// CIR-LABEL: cir.func {{.*}}@arg_ld2(
// CIR-NOT: fp_offset
// CIR: %[[ALIGN_MASK:.+]] = cir.const #cir.int<-32> : !s64i
// CIR: %[[ALIGNED:.+]] = cir.ptr_mask %{{.+}}, %[[ALIGN_MASK]]
// CIR: %[[MEM_STEP:.+]] = cir.const #cir.int<32> : !s32i
// CIR: cir.ptr_stride %[[ALIGNED]], %[[MEM_STEP]]
// CIR: %[[VPTR:.+]] = cir.cast bitcast %[[ALIGNED]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.vector<2 x !cir.long_double<!cir.f80>>>
// CIR: cir.load align(32) %[[VPTR]] : !cir.ptr<!cir.vector<2 x !cir.long_double<!cir.f80>>>, !cir.vector<2 x !cir.long_double<!cir.f80>>

// LLVM-LABEL: define dso_local <2 x x86_fp80> @arg_ld2(i32 noundef %{{[^,)]+}}, ...)
// LLVM-NOT: icmp
// LLVM: %[[ALIGNED:.+]] = call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -32)
// LLVM: load <2 x x86_fp80>, ptr %[[ALIGNED]], align 32

// Two elements take 32 bytes, which pass in memory below AVX.  The vector on
// its own is still returned directly, but a record holding it is returned
// through sret.
void take_ld2(ld2 v) {}

// CIR-SSE: cir.func {{.*}}@take_ld2(%arg0: !cir.ptr<!cir.vector<2 x !cir.long_double<!cir.f80>>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<2 x !cir.long_double<!cir.f80>>, llvm.noundef} loc({{[^)]+}}))
// CIR-AVX: cir.func {{.*}}@take_ld2(%arg0: !cir.vector<2 x !cir.long_double<!cir.f80>> {llvm.noundef} loc({{[^)]+}}))
// CIR-AVX512: cir.func {{.*}}@take_ld2(%arg0: !cir.vector<2 x !cir.long_double<!cir.f80>> {llvm.noundef} loc({{[^)]+}}))
// LLVM-SSE: define dso_local void @take_ld2(ptr noundef byval(<2 x x86_fp80>) align 32 %{{[^,)]+}})
// LLVM-AVX: define dso_local void @take_ld2(<2 x x86_fp80> noundef %{{[^,)]+}})
// LLVM-AVX512: define dso_local void @take_ld2(<2 x x86_fp80> noundef %{{[^,)]+}})

void call_ld2(ld2 v) { take_ld2(v); }

// CIR-SSE: cir.call @take_ld2(%{{.+}}) : (!cir.ptr<!cir.vector<2 x !cir.long_double<!cir.f80>>> {llvm.align = 32 : i64, llvm.byval = !cir.vector<2 x !cir.long_double<!cir.f80>>, llvm.noundef}) -> ()
// CIR-AVX: cir.call @take_ld2(%{{.+}}) : (!cir.vector<2 x !cir.long_double<!cir.f80>> {llvm.noundef}) -> ()
// CIR-AVX512: cir.call @take_ld2(%{{.+}}) : (!cir.vector<2 x !cir.long_double<!cir.f80>> {llvm.noundef}) -> ()
// LLVM-SSE: call void @take_ld2(ptr noundef byval(<2 x x86_fp80>) align 32 %{{[^,)]+}})
// LLVM-AVX: call void @take_ld2(<2 x x86_fp80> noundef %{{[^,)]+}})
// LLVM-AVX512: call void @take_ld2(<2 x x86_fp80> noundef %{{[^,)]+}})

ld2 ret_ld2(ld2 *p) { return *p; }

// CIR: cir.func {{.*}}@ret_ld2(%arg0: !cir.ptr<!cir.vector<2 x !cir.long_double<!cir.f80>>> {llvm.noundef} loc({{[^)]+}})) -> !cir.vector<2 x !cir.long_double<!cir.f80>>
// LLVM: define dso_local <2 x x86_fp80> @ret_ld2(ptr noundef %{{[^,)]+}})

ld2 call_ret_ld2(ld2 *p) { return ret_ld2(p); }

// CIR: cir.call @ret_ld2(%{{.+}}) : (!cir.ptr<!cir.vector<2 x !cir.long_double<!cir.f80>>> {llvm.noundef}) -> !cir.vector<2 x !cir.long_double<!cir.f80>>
// LLVM: call <2 x x86_fp80> @ret_ld2(ptr noundef %{{[^,)]+}})

struct S2 { ld2 v; };
struct S2 take_s2(struct S2 s) { return s; }

// CIR-SSE: cir.func {{.*}}@take_s2(%arg0: !cir.ptr<!rec_S2> {llvm.align = 32 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_S2, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_S2> {llvm.align = 32 : i64, llvm.byval = !rec_S2, llvm.noundef} loc({{[^)]+}}))
// CIR-AVX: cir.func {{.*}}@take_s2(%arg0: !cir.vector<2 x !cir.f80> loc({{[^)]+}})) -> !cir.vector<2 x !cir.f80>
// CIR-AVX512: cir.func {{.*}}@take_s2(%arg0: !cir.vector<2 x !cir.f80> loc({{[^)]+}})) -> !cir.vector<2 x !cir.f80>
// LLVM-SSE: define dso_local void @take_s2(ptr dead_on_unwind noalias writable sret(%struct.S2) align 32 %{{[^,)]+}}, ptr noundef byval(%struct.S2) align 32 %{{[^,)]+}})
// LLVM-AVX: define dso_local <2 x x86_fp80> @take_s2(<2 x x86_fp80> %{{[^,)]+}})
// LLVM-AVX512: define dso_local <2 x x86_fp80> @take_s2(<2 x x86_fp80> %{{[^,)]+}})

struct S2 call_s2(struct S2 s) { return take_s2(s); }

// CIR-SSE: cir.call @take_s2(%arg0, %{{.+}}) : (!cir.ptr<!rec_S2> {llvm.align = 32 : i64, llvm.dead_on_unwind, llvm.sret = !rec_S2, llvm.writable}, !cir.ptr<!rec_S2> {llvm.align = 32 : i64, llvm.byval = !rec_S2, llvm.noundef}) -> ()
// CIR-AVX: cir.call @take_s2(%{{.+}}) : (!cir.vector<2 x !cir.f80>) -> !cir.vector<2 x !cir.f80>
// CIR-AVX512: cir.call @take_s2(%{{.+}}) : (!cir.vector<2 x !cir.f80>) -> !cir.vector<2 x !cir.f80>
// LLVM-SSE: call void @take_s2(ptr dead_on_unwind writable sret(%struct.S2) align 32 %{{[^,)]+}}, ptr noundef byval(%struct.S2) align 32 %{{[^,)]+}})
// LLVM-AVX: call <2 x x86_fp80> @take_s2(<2 x x86_fp80> %{{[^,)]+}})
// LLVM-AVX512: call <2 x x86_fp80> @take_s2(<2 x x86_fp80> %{{[^,)]+}})

// Unnamed, a record wider than 16 bytes goes to memory even where a named one
// is passed in registers.
void var_s2(struct S2 s) { var(1, s); }

// CIR-SSE: cir.func {{.*}}@var_s2(%arg0: !cir.ptr<!rec_S2> {llvm.align = 32 : i64, llvm.byval = !rec_S2, llvm.noundef} loc({{[^)]+}}))
// CIR-AVX: cir.func {{.*}}@var_s2(%arg0: !cir.vector<2 x !cir.f80> loc({{[^)]+}}))
// CIR-AVX512: cir.func {{.*}}@var_s2(%arg0: !cir.vector<2 x !cir.f80> loc({{[^)]+}}))
// CIR: cir.call @var(%{{.+}}, %{{.+}}) : (!s32i {llvm.noundef}, !cir.ptr<!rec_S2> {llvm.align = 32 : i64, llvm.byval = !rec_S2, llvm.noundef}) -> ()
// LLVM-SSE: define dso_local void @var_s2(ptr noundef byval(%struct.S2) align 32 %{{[^,)]+}})
// LLVM-AVX: define dso_local void @var_s2(<2 x x86_fp80> %{{[^,)]+}})
// LLVM-AVX512: define dso_local void @var_s2(<2 x x86_fp80> %{{[^,)]+}})
// LLVM: call void (i32, ...) @var(i32 noundef 1, ptr noundef byval(%struct.S2) align 32 %{{[^,)]+}})

// Four elements take 64 bytes, which pass in registers only with AVX-512.
void take_ld4(ld4 v) {}

// CIR-SSE: cir.func {{.*}}@take_ld4(%arg0: !cir.ptr<!cir.vector<4 x !cir.long_double<!cir.f80>>> {llvm.align = 64 : i64, llvm.byval = !cir.vector<4 x !cir.long_double<!cir.f80>>, llvm.noundef} loc({{[^)]+}}))
// CIR-AVX: cir.func {{.*}}@take_ld4(%arg0: !cir.ptr<!cir.vector<4 x !cir.long_double<!cir.f80>>> {llvm.align = 64 : i64, llvm.byval = !cir.vector<4 x !cir.long_double<!cir.f80>>, llvm.noundef} loc({{[^)]+}}))
// CIR-AVX512: cir.func {{.*}}@take_ld4(%arg0: !cir.vector<4 x !cir.long_double<!cir.f80>> {llvm.noundef} loc({{[^)]+}}))
// LLVM-SSE: define dso_local void @take_ld4(ptr noundef byval(<4 x x86_fp80>) align 64 %{{[^,)]+}})
// LLVM-AVX: define dso_local void @take_ld4(ptr noundef byval(<4 x x86_fp80>) align 64 %{{[^,)]+}})
// LLVM-AVX512: define dso_local void @take_ld4(<4 x x86_fp80> noundef %{{[^,)]+}})

void call_ld4(ld4 v) { take_ld4(v); }

// CIR-SSE: cir.call @take_ld4(%{{.+}}) : (!cir.ptr<!cir.vector<4 x !cir.long_double<!cir.f80>>> {llvm.align = 64 : i64, llvm.byval = !cir.vector<4 x !cir.long_double<!cir.f80>>, llvm.noundef}) -> ()
// CIR-AVX: cir.call @take_ld4(%{{.+}}) : (!cir.ptr<!cir.vector<4 x !cir.long_double<!cir.f80>>> {llvm.align = 64 : i64, llvm.byval = !cir.vector<4 x !cir.long_double<!cir.f80>>, llvm.noundef}) -> ()
// CIR-AVX512: cir.call @take_ld4(%{{.+}}) : (!cir.vector<4 x !cir.long_double<!cir.f80>> {llvm.noundef}) -> ()
// LLVM-SSE: call void @take_ld4(ptr noundef byval(<4 x x86_fp80>) align 64 %{{[^,)]+}})
// LLVM-AVX: call void @take_ld4(ptr noundef byval(<4 x x86_fp80>) align 64 %{{[^,)]+}})
// LLVM-AVX512: call void @take_ld4(<4 x x86_fp80> noundef %{{[^,)]+}})

struct S4 { ld4 v; };
struct S4 take_s4(struct S4 s) { return s; }

// CIR-SSE: cir.func {{.*}}@take_s4(%arg0: !cir.ptr<!rec_S4> {llvm.align = 64 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_S4, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_S4> {llvm.align = 64 : i64, llvm.byval = !rec_S4, llvm.noundef} loc({{[^)]+}}))
// CIR-AVX: cir.func {{.*}}@take_s4(%arg0: !cir.ptr<!rec_S4> {llvm.align = 64 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_S4, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_S4> {llvm.align = 64 : i64, llvm.byval = !rec_S4, llvm.noundef} loc({{[^)]+}}))
// CIR-AVX512: cir.func {{.*}}@take_s4(%arg0: !cir.vector<4 x !cir.f80> loc({{[^)]+}})) -> !cir.vector<4 x !cir.f80>
// LLVM-SSE: define dso_local void @take_s4(ptr dead_on_unwind noalias writable sret(%struct.S4) align 64 %{{[^,)]+}}, ptr noundef byval(%struct.S4) align 64 %{{[^,)]+}})
// LLVM-AVX: define dso_local void @take_s4(ptr dead_on_unwind noalias writable sret(%struct.S4) align 64 %{{[^,)]+}}, ptr noundef byval(%struct.S4) align 64 %{{[^,)]+}})
// LLVM-AVX512: define dso_local <4 x x86_fp80> @take_s4(<4 x x86_fp80> %{{[^,)]+}})

struct S4 call_s4(struct S4 s) { return take_s4(s); }

// CIR-SSE: cir.call @take_s4(%arg0, %{{.+}}) : (!cir.ptr<!rec_S4> {llvm.align = 64 : i64, llvm.dead_on_unwind, llvm.sret = !rec_S4, llvm.writable}, !cir.ptr<!rec_S4> {llvm.align = 64 : i64, llvm.byval = !rec_S4, llvm.noundef}) -> ()
// CIR-AVX: cir.call @take_s4(%arg0, %{{.+}}) : (!cir.ptr<!rec_S4> {llvm.align = 64 : i64, llvm.dead_on_unwind, llvm.sret = !rec_S4, llvm.writable}, !cir.ptr<!rec_S4> {llvm.align = 64 : i64, llvm.byval = !rec_S4, llvm.noundef}) -> ()
// CIR-AVX512: cir.call @take_s4(%{{.+}}) : (!cir.vector<4 x !cir.f80>) -> !cir.vector<4 x !cir.f80>
// LLVM-SSE: call void @take_s4(ptr dead_on_unwind writable sret(%struct.S4) align 64 %{{[^,)]+}}, ptr noundef byval(%struct.S4) align 64 %{{[^,)]+}})
// LLVM-AVX: call void @take_s4(ptr dead_on_unwind writable sret(%struct.S4) align 64 %{{[^,)]+}}, ptr noundef byval(%struct.S4) align 64 %{{[^,)]+}})
// LLVM-AVX512: call <4 x x86_fp80> @take_s4(<4 x x86_fp80> %{{[^,)]+}})

// A long double stores its first 10 bytes, so it can share a union with
// members whose data ends within them, counting neither a record's tail
// padding nor an empty array.
struct __attribute__((aligned(16))) C16 { char c; };
union LDPad { long double ld; struct C16 s; };
union LDPad take_ldpad(union LDPad u) { return u; }

// CIR: cir.func {{.*}}@take_ldpad(%arg0: !cir.ptr<!rec_LDPad> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_LDPad, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_LDPad> {llvm.align = 16 : i64, llvm.byval = !rec_LDPad, llvm.noundef} loc({{[^)]+}}))
// LLVM: define dso_local void @take_ldpad(ptr dead_on_unwind noalias writable sret(%union.LDPad) align 16 %{{[^,)]+}}, ptr noundef byval(%union.LDPad) align 16 %{{[^,)]+}})

union LDZero { long double ld; int z[0]; };
union LDZero take_ldzero(union LDZero u) { return u; }

// CIR: cir.func {{.*}}@take_ldzero(%arg0: !cir.ptr<!rec_LDZero> {llvm.align = 16 : i64, llvm.byval = !rec_LDZero, llvm.noundef} loc({{[^)]+}})) -> !cir.f80
// LLVM: define dso_local x86_fp80 @take_ldzero(ptr noundef byval(%union.LDZero) align 16 %{{[^,)]+}})

// An empty array of long double holds no x87 value, so a union holding one is
// classified like any other.
union EmptyX87 { long l; long double z[0]; };
union EmptyX87 take_empty_x87(union EmptyX87 u) { return u; }

// CIR: cir.func {{.*}}@take_empty_x87(%arg0: !s64i loc({{[^)]+}})) -> !s64i
// LLVM: define dso_local i64 @take_empty_x87(i64 %{{[^,)]+}})

union LDTen { long double ld; char c[10]; };
union LDTen take_ldten(union LDTen u) { return u; }

// CIR: cir.func {{.*}}@take_ldten(%arg0: !u64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> !rec_anon_struct{{[0-9]*}}
// LLVM: define dso_local { i64, i64 } @take_ldten(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

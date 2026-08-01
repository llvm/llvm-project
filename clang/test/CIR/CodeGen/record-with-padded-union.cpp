// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

struct SSO {
  char *p;
  unsigned long len;
  union {
    char local[16];
    unsigned long capacity;
  };
};

// Inner union's tail padding must not bleed into the outer record.
// CIR: !rec_anon{{.*}} = !cir.union<"anon{{.*}}" {!cir.array<!s8i x 16>, !u64i}, padding = {!cir.array<!u8i x 8>}>
// CIR: !rec_SSO = !cir.struct<"SSO" {!cir.ptr<!s8i>, !u64i, !rec_anon{{.*}}}>

// LLVM: %struct.SSO = type { ptr, i64, %union.anon{{.*}} }
// LLVM: %union.anon{{.*}} = type { i64, [8 x i8] }

SSO s1 {nullptr, 1, {.local{}}};
// CIR: cir.global external @s1 = #cir.const_record<{#cir.ptr<null> : !cir.ptr<!s8i>, #cir.int<1> : !u64i, #cir.const_record<{#cir.zero : !cir.array<!s8i x 16>}> : !rec_anon2E0}> : !rec_SSO
// LLVM: @s1 = global { ptr, i64, { [16 x i8] } } { ptr null, i64 1, { [16 x i8] } zeroinitializer }
SSO s2 {nullptr, 1, {'a', 'b', 'c', 'd'}};
// CIR: cir.global external @s2 = #cir.const_record<{#cir.ptr<null> : !cir.ptr<!s8i>, #cir.int<1> : !u64i, #cir.const_record<{#cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i, #cir.int<100> : !s8i], trailing_zeros> : !cir.array<!s8i x 16>}> : !rec_anon2E0}> : !rec_SSO
// LLVMCIR: @s2 = global { ptr, i64, { [16 x i8] } } { ptr null, i64 1, { [16 x i8] } { [16 x i8] c"abcd\00\00\00\00\00\00\00\00\00\00\00\00" } }
// OGCG:    @s2 = global { ptr, i64, { <{ i8, i8, i8, i8, [12 x i8] }> } } { ptr null, i64 1, { <{ i8, i8, i8, i8, [12 x i8] }> } { <{ i8, i8, i8, i8, [12 x i8] }> <{ i8 97, i8 98, i8 99, i8 100, [12 x i8] zeroinitializer }> } }

struct SSO2 {
  char *p;
  union {
    char buf[16];
    long cap;
  };
  constexpr SSO2() : p(buf), buf{} {}
} str;
// CIR: cir.global external @str = #cir.const_record<{#cir.global_view<@str, [1 : i32]> : !cir.ptr<!s8i>, #cir.const_record<{#cir.zero : !cir.array<!s8i x 16>}> : !rec_anon2E1}> : !rec_SSO2
// LLVM: @str = global { ptr, { [16 x i8] } } { ptr getelementptr {{.*}}(i8, ptr @str, i64 8), { [16 x i8] } zeroinitializer }

extern "C" SSO *last_of_three() {
  SSO *p = new SSO[3];
  return &p[2];
}

// Allocation is 3*sizeof(SSO)=96; per-element stride comes from struct size.
// LLVM-LABEL: define {{.*}}@last_of_three
// LLVM: call {{.*}}@_Znam(i64 noundef 96)
// LLVM: getelementptr{{.*}}%struct.SSO, ptr %{{.+}}, i64 2



// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// _BitInt types are distinguished from regular integer types via the
// "bitint" keyword in CIR.  Verify that the type alias includes "_bitint"
// and that regular __int128 does not.

// CIR-DAG: !s32i_bitint = !cir.int<s, 32, bitint>
// CIR-DAG: !s128i_bitint = !cir.int<s, 128, bitint>
// CIR-DAG: !u64i_bitint = !cir.int<u, 64, bitint>
// CIR-DAG: !s128i = !cir.int<s, 128>

// _BitInt(128) has alignment 8 while __int128 has alignment 16.
signed _BitInt(128) bitint128_var;
__int128 int128_var;

// CIR: cir.global external @bitint128_var = #cir.int<0> : !s128i_bitint {alignment = 8 : i64}
// CIR: cir.global external @int128_var = #cir.int<0> : !s128i {alignment = 16 : i64}

// LLVM: @bitint128_var = global i128 0, align 8
// LLVM: @int128_var = global i128 0, align 16

// OGCG: @bitint128_var = global i128 0, align 8
// OGCG: @int128_var = global i128 0, align 16

void take_bitint_32(_BitInt(32) x) {}
// CIR: cir.func {{.*}} @take_bitint_32(%arg0: !s32i_bitint
// LLVM: define {{.*}} void @take_bitint_32(i32 {{.*}})
// OGCG: define {{.*}} void @take_bitint_32(i32 {{.*}})

void take_bitint_128(signed _BitInt(128) x) {}
// CIR: cir.func {{.*}} @take_bitint_128(%arg0: !s128i_bitint
// LLVM: define {{.*}} void @take_bitint_128(i128 {{.*}})
// OGCG: define {{.*}} void @take_bitint_128(i128 {{.*}})

void take_unsigned_bitint(unsigned _BitInt(64) x) {}
// CIR: cir.func {{.*}} @take_unsigned_bitint(%arg0: !u64i_bitint
// LLVM: define {{.*}} void @take_unsigned_bitint(i64 {{.*}})
// OGCG: define {{.*}} void @take_unsigned_bitint(i64 {{.*}})

// Regular __int128 should NOT have the bitint flag.
void take_int128(__int128 x) {}
// CIR: cir.func {{.*}} @take_int128(%arg0: !s128i
// LLVM: define {{.*}} void @take_int128(i128 {{.*}})
// OGCG: define {{.*}} void @take_int128(i128 {{.*}})

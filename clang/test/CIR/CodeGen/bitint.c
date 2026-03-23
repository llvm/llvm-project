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

void take_bitint_32(_BitInt(32) x) {}
// LLVM: define {{.*}} void @take_bitint_32(i32 {{.*}})
// OGCG: define {{.*}} void @take_bitint_32(i32 {{.*}})

void take_bitint_128(signed _BitInt(128) x) {}
// LLVM: define {{.*}} void @take_bitint_128(i128 {{.*}})
// OGCG: define {{.*}} void @take_bitint_128(i128 {{.*}})

void take_unsigned_bitint(unsigned _BitInt(64) x) {}
// LLVM: define {{.*}} void @take_unsigned_bitint(i64 {{.*}})
// OGCG: define {{.*}} void @take_unsigned_bitint(i64 {{.*}})

// Regular __int128 should NOT have the bitint flag.
void take_int128(__int128 x) {}
// LLVM: define {{.*}} void @take_int128(i128 {{.*}})
// OGCG: define {{.*}} void @take_int128(i128 {{.*}})

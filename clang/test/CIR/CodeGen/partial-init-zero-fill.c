// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Globals where the brace-enclosed initializer covers fewer fields than the
// record has. Per C semantics the omitted tail / hole is zero-initialized.
// The constant emitter has to materialize those zeros in the const_record,
// otherwise the resulting global has undef bytes where the standard requires
// zero.

// Trailing fields zero-initialized.
struct PartialA { int a, b, c, d; };
struct PartialA pa = { 1 };

// CIR: cir.global external @pa
// CIR-SAME: #cir.int<1> : !s32i
// CIR-SAME: #cir.int<0> : !s32i
// CIR-SAME: #cir.int<0> : !s32i
// CIR-SAME: #cir.int<0> : !s32i
// LLVM: @pa ={{.*}}i32 1, i32 0, i32 0, i32 0
// OGCG: @pa ={{.*}}i32 1, i32 0, i32 0, i32 0

// Partial init of a record with array and trailing scalar. Everything past
// the first init slot must be zero.
struct PartialB { int a; double arr[4]; char tail; };
struct PartialB pb = { 7 };

// LLVM: @pb ={{.*}}i32 7
// LLVM-SAME: [4 x double] zeroinitializer
// LLVM-SAME: i8 0
// OGCG: @pb ={{.*}}i32 7
// OGCG-SAME: [4 x double] zeroinitializer
// OGCG-SAME: i8 0

// Hole in the middle: designated init skips an interior aggregate field.
// The skipped struct member must be zero-initialized in the const_record.
struct InnerC { int x, y; };
struct PartialC { int a; struct InnerC inner; int b; };
struct PartialC pc = { .b = 99 };

// LLVM: @pc ={{.*}}i32 0
// LLVM-SAME: zeroinitializer
// LLVM-SAME: i32 99
// OGCG: @pc ={{.*}}i32 0
// OGCG-SAME: zeroinitializer
// OGCG-SAME: i32 99

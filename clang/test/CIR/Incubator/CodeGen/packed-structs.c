// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#pragma pack(1)

typedef struct {
    int  a0;
    char a1;
} A;

typedef struct {
    int  b0;
    char b1;
    A a[6];
} B;

typedef struct {
    int  c0;
    char c1;
} __attribute__((aligned(2))) C;


// CIR: !rec_A = !cir.record<struct "A" packed {!s32i, !s8i}>
// CIR: !rec_C = !cir.record<struct "C" packed padded {!s32i, !s8i, !u8i}>
// CIR: !rec_D = !cir.record<struct "D" packed padded {!s8i, !u8i, !s32i}
// CIR: !rec_F = !cir.record<struct "F" packed {!s64i, !s8i}
// CIR: !rec_E = !cir.record<struct "E" packed {!rec_D
// CIR: !rec_G = !cir.record<struct "G" {!rec_F
// CIR: !rec_H = !cir.record<struct "H" {!s32i, !rec_anon2E0
// CIR: !rec_B = !cir.record<struct "B" packed {!s32i, !s8i, !cir.array<!rec_A x 6>}>
// CIR: !rec_I = !cir.record<struct "I" packed {!s8i, !rec_H
// CIR: !rec_J = !cir.record<struct "J" packed {!s8i, !s8i, !s8i, !s8i, !rec_I

// LLVM: %struct.A = type <{ i32, i8 }>
// LLVM: %struct.B = type <{ i32, i8, [6 x %struct.A] }>
// LLVM: %struct.C = type <{ i32, i8, i8 }>
// LLVM: %struct.E = type <{ %struct.D, i32 }>
// LLVM: %struct.D = type <{ i8, i8, i32 }>
// LLVM: %struct.G = type { %struct.F, i8 }
// LLVM: %struct.F = type <{ i64, i8 }>
// LLVM: %struct.J = type <{ i8, i8, i8, i8, %struct.I, i32 }>
// LLVM: %struct.I = type <{ i8, %struct.H }>
// LLVM: %struct.H = type { i32, %union.anon.{{.*}} }

// CIR: cir.func {{.*@foo()}}
// CIR:  {{.*}} = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a"] {alignment = 1 : i64}
// CIR:  {{.*}} = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["b"] {alignment = 1 : i64}
// CIR:  {{.*}} = cir.alloca !rec_C, !cir.ptr<!rec_C>, ["c"] {alignment = 2 : i64}

// LLVM: {{.*}} = alloca %struct.A, i64 1, align 1
// LLVM: {{.*}} = alloca %struct.B, i64 1, align 1
// LLVM: {{.*}} = alloca %struct.C, i64 1, align 2
void foo() {
    A a;
    B b;
    C c;
}

#pragma pack(2)

typedef struct {
    char b;
    int c;
} D;

typedef struct {
    D e;
    int f;
} E;

// CIR: cir.func {{.*@f1()}}
// CIR:  %[[E:.*]] = cir.alloca !rec_E, !cir.ptr<!rec_E>, ["a", init] {alignment = 2 : i64}
// CIR:  %[[ZERO:.*]] = cir.const #cir.zero : !rec_E
// CIR:  cir.store{{.*}} %[[ZERO]], %[[E]] : !rec_E, !cir.ptr<!rec_E>

// LLVM: {{.*}} = alloca %struct.E, i64 1, align 2
void f1() {
    E a = {};
}

#pragma pack(1)

typedef struct {
    long b;
    char c;
} F;

typedef struct {
    F e;
    char f;
} G;

// CIR: cir.func {{.*@f2()}}
// CIR:  {{.*}} = cir.alloca !rec_G, !cir.ptr<!rec_G>, ["a", init] {alignment = 1 : i64}

// LLVM: {{.*}} = alloca %struct.G, i64 1, align 1
void f2() {
    G a = {};
}

#pragma pack(1)

typedef struct {
    int d0;
    union {
        char null;
        int val;
    } value;
} H;

typedef struct {
    char t;
    H d;
} I;

typedef struct {
    char a0;
    char a1;
    char a2;
    char a3;
    I c;
    int a;
} J;

// CIR: cir.func {{.*@f3()}}
// CIR:  {{.*}} = cir.alloca !rec_J, !cir.ptr<!rec_J>, ["a", init] {alignment = 1 : i64}

// LLVM: {{.*}} = alloca %struct.J, i64 1, align 1
void f3() {
    J a = {0};
}

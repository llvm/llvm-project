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


// CIR: !ty_A = !cir.struct<struct "A" packed {!cir.int<s, 32>, !cir.int<s, 8>}>
// CIR: !ty_C = !cir.struct<struct "C" packed {!cir.int<s, 32>, !cir.int<s, 8>, !cir.int<u, 8>}>
// CIR: !ty_D = !cir.struct<struct "D" packed {!cir.int<s, 8>, !cir.int<u, 8>, !cir.int<s, 32>}
// CIR: !ty_F = !cir.struct<struct "F" packed {!cir.int<s, 64>, !cir.int<s, 8>}
// CIR: !ty_E = !cir.struct<struct "E" packed {!cir.struct<struct "D" packed {!cir.int<s, 8>, !cir.int<u, 8>, !cir.int<s, 32>}
// CIR: !ty_G = !cir.struct<struct "G" {!cir.struct<struct "F" packed {!cir.int<s, 64>, !cir.int<s, 8>}
// CIR: !ty_H = !cir.struct<struct "H" {!cir.int<s, 32>, !cir.struct<union "anon.{{.*}}" {!cir.int<s, 8>, !cir.int<s, 32>}
// CIR: !ty_B = !cir.struct<struct "B" packed {!cir.int<s, 32>, !cir.int<s, 8>, !cir.array<!cir.struct<struct "A" packed {!cir.int<s, 32>, !cir.int<s, 8>}> x 6>}>
// CIR: !ty_I = !cir.struct<struct "I" packed {!cir.int<s, 8>, !cir.struct<struct "H" {!cir.int<s, 32>, !cir.struct<union "anon.{{.*}}" {!cir.int<s, 8>, !cir.int<s, 32>}
// CIR: !ty_J = !cir.struct<struct "J" packed {!cir.int<s, 8>, !cir.int<s, 8>, !cir.int<s, 8>, !cir.int<s, 8>, !cir.struct<struct "I" packed {!cir.int<s, 8>, !cir.struct<struct "H" {!cir.int<s, 32>, !cir.struct<union "anon.{{.*}}" {!cir.int<s, 8>, !cir.int<s, 32>}

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
// CIR:  {{.*}} = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["a"] {alignment = 1 : i64}
// CIR:  {{.*}} = cir.alloca !ty_B, !cir.ptr<!ty_B>, ["b"] {alignment = 1 : i64}
// CIR:  {{.*}} = cir.alloca !ty_C, !cir.ptr<!ty_C>, ["c"] {alignment = 2 : i64}

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
// CIR:  {{.*}} = cir.alloca !ty_E, !cir.ptr<!ty_E>, ["a"] {alignment = 2 : i64}

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
// CIR:  {{.*}} = cir.alloca !ty_G, !cir.ptr<!ty_G>, ["a"] {alignment = 1 : i64}

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
// CIR:  {{.*}} = cir.alloca !ty_J, !cir.ptr<!ty_J>, ["a"] {alignment = 1 : i64}

// LLVM: {{.*}} = alloca %struct.J, i64 1, align 1
void f3() {
    J a = {0};
}
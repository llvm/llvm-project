// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

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


// CHECK: !ty_A = !cir.struct<struct "A" packed {!cir.int<s, 32>, !cir.int<s, 8>}>
// CHECK: !ty_C = !cir.struct<struct "C" packed {!cir.int<s, 32>, !cir.int<s, 8>}>
// CHECK: !ty_B = !cir.struct<struct "B" packed {!cir.int<s, 32>, !cir.int<s, 8>, !cir.array<!cir.struct<struct "A" packed {!cir.int<s, 32>, !cir.int<s, 8>}> x 6>}>

// CHECK: cir.func {{.*@foo()}}
// CHECK:  %0 = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["a"] {alignment = 1 : i64}
// CHECK:  %1 = cir.alloca !ty_B, !cir.ptr<!ty_B>, ["b"] {alignment = 1 : i64}
// CHECK:  %2 = cir.alloca !ty_C, !cir.ptr<!ty_C>, ["c"] {alignment = 2 : i64}
void foo() {
    A a;
    B b;
    C c;
}



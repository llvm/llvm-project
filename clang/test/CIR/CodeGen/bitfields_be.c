// RUN: %clang_cc1 -triple aarch64_be-unknown-linux-gnu -emit-llvm %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=LLVM

// RUN: %clang_cc1 -triple aarch64_be-unknown-linux-gnu -fclangir -emit-llvm %s -o %t1.cir
// RUN: FileCheck --input-file=%t1.cir %s 

typedef struct {
    int a : 4;
    int b : 11;
    int c : 17;
} S;

void init(S* s) {
    s->a = -4;
    s->b = 42;
    s->c = -12345;
}

// field 'a'
// LLVM:   %[[PTR0:.*]] = load ptr
// CHECK:  %[[PTR0:.*]] = load ptr
// LLVM:   %[[VAL0:.*]] = load i32, ptr %[[PTR0]]
// CHECK:  %[[VAL0:.*]] = load i32, ptr %[[PTR0]]
// LLVM:   %[[AND0:.*]] = and i32 %[[VAL0]], 268435455
// CHECK:  %[[AND0:.*]] = and i32 %[[VAL0]], 268435455
// LLVM:   %[[OR0:.*]] = or i32 %[[AND0]], -1073741824
// CHECK:  %[[OR0:.*]] = or i32 %[[AND0]], -1073741824
// LLVM:   store i32 %[[OR0]], ptr %[[PTR0]]
// CHECK:  store i32 %[[OR0]], ptr %[[PTR0]]

// field 'b'
// LLVM:   %[[PTR1:.*]] = load ptr
// CHECK:  %[[PTR1:.*]] = load ptr
// LLVM:   %[[VAL1:.*]] = load i32, ptr %[[PTR1]]
// CHECK:  %[[VAL1:.*]] = load i32, ptr %[[PTR1]]
// LLVM:   %[[AND1:.*]] = and i32 %[[VAL1]], -268304385
// CHECK:  %[[AND1:.*]] = and i32 %[[VAL1]], -268304385
// LLVM:   %[[OR1:.*]] = or i32 %[[AND1]], 5505024
// CHECK:  %[[OR1:.*]] = or i32 %[[AND1]], 5505024
// LLVM:   store i32 %[[OR1]], ptr %[[PTR1]]
// CHECK:  store i32 %[[OR1]], ptr %[[PTR1]]

// field 'c'
// LLVM:   %[[PTR2:.*]] = load ptr
// CHECK:  %[[PTR2:.*]] = load ptr
// LLVM:   %[[VAL2:.*]] = load i32, ptr %[[PTR2]]
// CHECK:  %[[VAL2:.*]] = load i32, ptr %[[PTR2]]
// LLVM:   %[[AND2:.*]] = and i32 %[[VAL2]], -131072
// CHECK:  %[[AND2:.*]] = and i32 %[[VAL2]], -131072
// LLVM:   %[[OR2:.*]] = or i32 %[[AND2]], 118727
// CHECK:  %[[OR2:.*]] = or i32 %[[AND2]], 118727
// LLVM:   store i32 %[[OR2]], ptr %[[PTR2]]
// CHECK:  store i32 %[[OR2]], ptr %[[PTR2]]


// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int foo(void) { return __extension__ 0b101010; }

//CHECK: cir.func @foo() -> !s32i extra( {inline = #cir.inline<no>, optnone = #cir.optnone} ) {
//CHECK-NEXT:    [[ADDR:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
//CHECK-NEXT:    [[VAL:%.*]] = cir.const(#cir.int<42> : !s32i) : !s32i
//CHECK-NEXT:    cir.store [[VAL]], [[ADDR]] : !s32i, cir.ptr <!s32i>
//CHECK-NEXT:    [[LOAD_VAL:%.*]] = cir.load [[ADDR]] : cir.ptr <!s32i>, !s32i
//CHECK-NEXT:    cir.return [[LOAD_VAL]] : !s32i

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct __attribute__((packed)) PackedStruct {
    char a1;
    char a2;
    char a3;
};
struct PackedStruct packed[10];
char *packed_element = &(packed[-2].a3);
// CHECK: cir.global  external @packed = #cir.zero : !cir.array<!ty_PackedStruct x 10> {alignment = 16 : i64} loc(#loc5)
// CHECK: cir.global  external @packed_element = #cir.global_view<@packed, [-2 : i32, 2 : i32]>
// LLVM: @packed = global [10 x %struct.PackedStruct] zeroinitializer
// LLVM: @packed_element = global ptr getelementptr inbounds ([10 x %struct.PackedStruct], ptr @packed, i32 -2, i32 2)

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s --check-prefix=MLIR

int test_array1() {
    // CIR-LABEL: cir.func {{.*}} @test_array1
    // CIR: %[[ARRAY:.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["a"] {alignment = 4 : i64}
    // CIR: %{{.*}} = cir.cast(array_to_ptrdecay, %[[ARRAY]] : !cir.ptr<!cir.array<!s32i x 3>>), !cir.ptr<!s32i>

    // MLIR-LABEL: func @test_array1
    // MLIR: %{{.*}} = memref.alloca() {alignment = 4 : i64} : memref<i32>
    // MLIR: %[[ARRAY:.*]] = memref.alloca() {alignment = 4 : i64} : memref<3xi32>
    // MLIR: %{{.*}} = memref.load %[[ARRAY]][%{{.*}}] : memref<3xi32>
    int a[3];
    return a[1]; 
}

int test_array2() {
    // CIR-LABEL: cir.func {{.*}} @test_array2
    // CIR: %[[ARRAY:.*]] = cir.alloca !cir.array<!cir.array<!s32i x 4> x 3>, !cir.ptr<!cir.array<!cir.array<!s32i x 4> x 3>>, ["a"] {alignment = 16 : i64}
    // CIR: %{{.*}} = cir.cast(array_to_ptrdecay, %[[ARRAY]] : !cir.ptr<!cir.array<!cir.array<!s32i x 4> x 3>>), !cir.ptr<!cir.array<!s32i x 4>>
    // CIR: %{{.*}} = cir.cast(array_to_ptrdecay, %{{.*}} : !cir.ptr<!cir.array<!s32i x 4>>), !cir.ptr<!s32i>

    // MLIR-LABEL: func @test_array2
    // MLIR: %{{.*}} = memref.alloca() {alignment = 4 : i64} : memref<i32>
    // MLIR: %[[ARRAY:.*]] = memref.alloca() {alignment = 16 : i64} : memref<3x4xi32>
    // MLIR: %{{.*}} = memref.load %[[ARRAY]][%{{.*}}, %{{.*}}] : memref<3x4xi32>
    int a[3][4];
    return a[1][2]; 
}

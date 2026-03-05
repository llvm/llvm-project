// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s --check-prefix=MLIR

int test_array1() {
    // CIR-LABEL: cir.func {{.*}} @test_array1
    // CIR: %[[ARRAY:.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["a"] {alignment = 4 : i64}
    // CIR: %{{.*}} = cir.get_element %[[ARRAY]][{{.*}}] : (!cir.ptr<!cir.array<!s32i x 3>>, !s32i) -> !cir.ptr<!s32i>

    // MLIR-LABEL: func @test_array1
    // MLIR: %{{.*}} = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
    // MLIR: %[[ARRAY:.*]] = memref.alloca() {alignment = 4 : i64} : memref<3xi32>
    // MLIR: %{{.*}} = memref.load %[[ARRAY]][%{{.*}}] : memref<3xi32>
    int a[3];
    return a[1];
}

int test_array2() {
    // CIR-LABEL: cir.func {{.*}} @test_array2
    // CIR: %[[ARRAY:.*]] = cir.alloca !cir.array<!cir.array<!s32i x 4> x 3>, !cir.ptr<!cir.array<!cir.array<!s32i x 4> x 3>>, ["a"] {alignment = 16 : i64}
    // CIR: %{{.*}} = cir.get_element %[[ARRAY]][%{{.*}}] : (!cir.ptr<!cir.array<!cir.array<!s32i x 4> x 3>>, !s32i) -> !cir.ptr<!cir.array<!s32i x 4>>
    // CIR: %{{.*}} = cir.get_element %{{.*}}[%{{.*}}] : (!cir.ptr<!cir.array<!s32i x 4>>, !s32i) -> !cir.ptr<!s32i>

    // MLIR-LABEL: func @test_array2
    // MLIR: %{{.*}} = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
    // MLIR: %[[ARRAY:.*]] = memref.alloca() {alignment = 16 : i64} : memref<3x4xi32>
    // MLIR: %{{.*}} = memref.load %[[ARRAY]][%{{.*}}, %{{.*}}] : memref<3x4xi32>
    int a[3][4];
    return a[1][2];
}

int test_array3() {
    // CIR-LABEL: cir.func {{.*}} @test_array3()
    // CIR: %[[ARRAY:.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["a"] {alignment = 4 : i64}
    // CIR: %[[ELEM1:.*]] = cir.get_element %[[ARRAY]][{{.*}}] : (!cir.ptr<!cir.array<!s32i x 3>>, !s32i) -> !cir.ptr<!s32i>
    // CIR: {{.*}} = cir.load align(4) %[[ELEM1]] : !cir.ptr<!s32i>, !s32i
    // CIR: %[[ELEM2:.*]] = cir.get_element %[[ARRAY]][{{.*}}] : (!cir.ptr<!cir.array<!s32i x 3>>, !s32i) -> !cir.ptr<!s32i>
    // CIR: %{{.*}} = cir.load align(4) %[[ELEM2]] : !cir.ptr<!s32i>, !s32i
    // CIR: cir.store align(4) {{.*}}, %[[ELEM2]] : !s32i, !cir.ptr<!s32i>
    // CIR: %[[ELEM3:.*]] = cir.get_element %[[ARRAY]][{{.*}}] : (!cir.ptr<!cir.array<!s32i x 3>>, !s32i) -> !cir.ptr<!s32i>
    // CIR: %{{.*}} = cir.load align(4) %[[ELEM3]] : !cir.ptr<!s32i>, !s32i

    // MLIR-LABEL: func @test_array3
    // MLIR: %{{.*}} = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
    // MLIR: %[[ARRAY:.*]] = memref.alloca() {alignment = 4 : i64} : memref<3xi32>
    // MLIR: %[[IDX1:.*]] = arith.index_cast %{{.*}} : i32 to index
    // MLIR: %{{.*}} = memref.load %[[ARRAY]][%[[IDX1]]] : memref<3xi32>
    // MLIR: %[[IDX2:.*]] = arith.index_cast %{{.*}} : i32 to index
    // MLIR: %{{.*}} = memref.load %[[ARRAY]][%[[IDX2]]] : memref<3xi32>
    // MLIR: memref.store %{{.*}}, %[[ARRAY]][%[[IDX2]]] : memref<3xi32>
    // MLIR: %[[IDX3:.*]] = arith.index_cast %{{.*}} : i32 to index
    // MLIR: %{{.*}} = memref.load %[[ARRAY]][%[[IDX3]]] : memref<3xi32>
    int a[3];
    a[0] += a[2];
    return a[1];
}

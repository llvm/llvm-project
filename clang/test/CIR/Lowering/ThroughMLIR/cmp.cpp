// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

bool testSignedIntCmpOps(int a, int b) {
    // CHECK: %[[ALLOC1:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
    // CHECK: %[[ALLOC2:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
    // CHECK: %[[ALLOC3:.+]] = memref.alloca() {alignment = 1 : i64} : memref<i8>
    // CHECK: %[[ALLOC4:.+]] = memref.alloca() {alignment = 1 : i64} : memref<i8>
    // CHECK: memref.store %arg0, %[[ALLOC1]][] : memref<i32>
    // CHECK: memref.store %arg1, %[[ALLOC2]][] : memref<i32>  
  
    bool x = a == b;
  
    // CHECK: %[[LOAD0:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD1:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP0:.+]] = arith.cmpi eq, %[[LOAD0]], %[[LOAD1]] : i32
    // CHECK: %[[EXT0:.+]] = arith.extui %[[CMP0]] : i1 to i8
    // CHECK: memref.store %[[EXT0]], %[[ALLOC4]][] : memref<i8>

    x = a != b;

    // CHECK: %[[LOAD2:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD3:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP1:.+]] = arith.cmpi ne, %[[LOAD2]], %[[LOAD3]] : i32
    // CHECK: %[[EXT1:.+]] = arith.extui %[[CMP1]] : i1 to i8
    // CHECK: memref.store %[[EXT1]], %[[ALLOC4]][] : memref<i8>

    x = a < b;

    // CHECK: %[[LOAD4:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD5:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP2:.+]] = arith.cmpi slt, %[[LOAD4]], %[[LOAD5]] : i32
    // CHECK: %[[EXT2:.+]] = arith.extui %[[CMP2]] : i1 to i8
    // CHECK: memref.store %[[EXT2]], %[[ALLOC4]][] : memref<i8>

    x = a <= b;

    // CHECK: %[[LOAD6:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD7:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP3:.+]] = arith.cmpi sle, %[[LOAD6]], %[[LOAD7]] : i32
    // CHECK: %[[EXT3:.+]] = arith.extui %[[CMP3]] : i1 to i8
    // CHECK: memref.store %[[EXT3]], %[[ALLOC4]][] : memref<i8>

    x = a > b;

    // CHECK: %[[LOAD8:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD9:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP4:.+]] = arith.cmpi sgt, %[[LOAD8]], %[[LOAD9]] : i32
    // CHECK: %[[EXT4:.+]] = arith.extui %[[CMP4]] : i1 to i8
    // CHECK: memref.store %[[EXT4]], %[[ALLOC4]][] : memref<i8>

    x = a >= b;

    // CHECK: %[[LOAD10:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD11:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP5:.+]] = arith.cmpi sge, %[[LOAD10]], %[[LOAD11]] : i32
    // CHECK: %[[EXT5:.+]] = arith.extui %[[CMP5]] : i1 to i8
    // CHECK: memref.store %[[EXT5]], %[[ALLOC4]][] : memref<i8>

    // CHECK: %[[LOAD12:.+]] = memref.load %[[ALLOC4]][] : memref<i8>
    // CHECK: memref.store %[[LOAD12]], %[[ALLOC3]][] : memref<i8>
    // CHECK: %[[LOAD13:.+]] = memref.load %[[ALLOC3]][] : memref<i8>
    // CHECK: return %[[LOAD13]] : i8
    return x;
}

bool testUnSignedIntBinOps(unsigned a, unsigned b) {
    // CHECK: %[[ALLOC1:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
    // CHECK: %[[ALLOC2:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
    // CHECK: %[[ALLOC3:.+]] = memref.alloca() {alignment = 1 : i64} : memref<i8>
    // CHECK: %[[ALLOC4:.+]] = memref.alloca() {alignment = 1 : i64} : memref<i8>
    // CHECK: memref.store %arg0, %[[ALLOC1]][] : memref<i32>
    // CHECK: memref.store %arg1, %[[ALLOC2]][] : memref<i32>
    
    bool x = a == b;

    // CHECK: %[[LOAD0:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD1:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP0:.+]] = arith.cmpi eq, %[[LOAD0]], %[[LOAD1]] : i32
    // CHECK: %[[EXT0:.+]] = arith.extui %[[CMP0]] : i1 to i8
    // CHECK: memref.store %[[EXT0]], %[[ALLOC4]][] : memref<i8>

    x = a != b;

    // CHECK: %[[LOAD2:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD3:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP1:.+]] = arith.cmpi ne, %[[LOAD2]], %[[LOAD3]] : i32
    // CHECK: %[[EXT1:.+]] = arith.extui %[[CMP1]] : i1 to i8
    // CHECK: memref.store %[[EXT1]], %[[ALLOC4]][] : memref<i8>

    x = a < b;

    // CHECK: %[[LOAD4:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD5:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP2:.+]] = arith.cmpi ult, %[[LOAD4]], %[[LOAD5]] : i32
    // CHECK: %[[EXT2:.+]] = arith.extui %[[CMP2]] : i1 to i8
    // CHECK: memref.store %[[EXT2]], %[[ALLOC4]][] : memref<i8>

    x = a <= b;

    // CHECK: %[[LOAD6:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD7:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP3:.+]] = arith.cmpi ule, %[[LOAD6]], %[[LOAD7]] : i32
    // CHECK: %[[EXT3:.+]] = arith.extui %[[CMP3]] : i1 to i8
    // CHECK: memref.store %[[EXT3]], %[[ALLOC4]][] : memref<i8>

    x = a > b;

    // CHECK: %[[LOAD8:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD9:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP4:.+]] = arith.cmpi ugt, %[[LOAD8]], %[[LOAD9]] : i32
    // CHECK: %[[EXT4:.+]] = arith.extui %[[CMP4]] : i1 to i8
    // CHECK: memref.store %[[EXT4]], %[[ALLOC4]][] : memref<i8>

    x = a >= b;

    // CHECK: %[[LOAD10:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
    // CHECK: %[[LOAD11:.+]] = memref.load %[[ALLOC2]][] : memref<i32>
    // CHECK: %[[CMP5:.+]] = arith.cmpi uge, %[[LOAD10]], %[[LOAD11]] : i32
    // CHECK: %[[EXT5:.+]] = arith.extui %[[CMP5]] : i1 to i8
    // CHECK: memref.store %[[EXT5]], %[[ALLOC4]][] : memref<i8>

    return x;
    // CHECK: return
}

bool testFloatingPointCmpOps(float a, float b) {
    // CHECK: %[[ALLOC1:.+]] = memref.alloca() {alignment = 4 : i64} : memref<f32>
    // CHECK: %[[ALLOC2:.+]] = memref.alloca() {alignment = 4 : i64} : memref<f32>
    // CHECK: %[[ALLOC3:.+]] = memref.alloca() {alignment = 1 : i64} : memref<i8>
    // CHECK: %[[ALLOC4:.+]] = memref.alloca() {alignment = 1 : i64} : memref<i8>
    // CHECK: memref.store %arg0, %[[ALLOC1]][] : memref<f32>
    // CHECK: memref.store %arg1, %[[ALLOC2]][] : memref<f32>

    bool x = a == b;

    // CHECK: %[[LOAD0:.+]] = memref.load %[[ALLOC1]][] : memref<f32>
    // CHECK: %[[LOAD1:.+]] = memref.load %[[ALLOC2]][] : memref<f32>
    // CHECK: %[[CMP0:.+]] = arith.cmpf oeq, %[[LOAD0]], %[[LOAD1]] : f32
    // CHECK: %[[EXT0:.+]] = arith.extui %[[CMP0]] : i1 to i8
    // CHECK: memref.store %[[EXT0]], %[[ALLOC4]][] : memref<i8>

    x = a != b;

    // CHECK: %[[LOAD2:.+]] = memref.load %[[ALLOC1]][] : memref<f32>
    // CHECK: %[[LOAD3:.+]] = memref.load %[[ALLOC2]][] : memref<f32>
    // CHECK: %[[CMP1:.+]] = arith.cmpf une, %[[LOAD2]], %[[LOAD3]] : f32
    // CHECK: %[[EXT1:.+]] = arith.extui %[[CMP1]] : i1 to i8
    // CHECK: memref.store %[[EXT1]], %[[ALLOC4]][] : memref<i8>

    x = a < b;

    // CHECK: %[[LOAD4:.+]] = memref.load %[[ALLOC1]][] : memref<f32>
    // CHECK: %[[LOAD5:.+]] = memref.load %[[ALLOC2]][] : memref<f32>
    // CHECK: %[[CMP2:.+]] = arith.cmpf olt, %[[LOAD4]], %[[LOAD5]] : f32
    // CHECK: %[[EXT2:.+]] = arith.extui %[[CMP2]] : i1 to i8
    // CHECK: memref.store %[[EXT2]], %[[ALLOC4]][] : memref<i8>

    x = a <= b;

    // CHECK: %[[LOAD6:.+]] = memref.load %[[ALLOC1]][] : memref<f32>
    // CHECK: %[[LOAD7:.+]] = memref.load %[[ALLOC2]][] : memref<f32>
    // CHECK: %[[CMP3:.+]] = arith.cmpf ole, %[[LOAD6]], %[[LOAD7]] : f32
    // CHECK: %[[EXT3:.+]] = arith.extui %[[CMP3]] : i1 to i8
    // CHECK: memref.store %[[EXT3]], %[[ALLOC4]][] : memref<i8>

    x = a > b;

    // CHECK: %[[LOAD8:.+]] = memref.load %[[ALLOC1]][] : memref<f32>
    // CHECK: %[[LOAD9:.+]] = memref.load %[[ALLOC2]][] : memref<f32>
    // CHECK: %[[CMP4:.+]] = arith.cmpf ogt, %[[LOAD8]], %[[LOAD9]] : f32
    // CHECK: %[[EXT4:.+]] = arith.extui %[[CMP4]] : i1 to i8
    // CHECK: memref.store %[[EXT4]], %[[ALLOC4]][] : memref<i8>

    x = a >= b;

    // CHECK: %[[LOAD10:.+]] = memref.load %[[ALLOC1]][] : memref<f32>
    // CHECK: %[[LOAD11:.+]] = memref.load %[[ALLOC2]][] : memref<f32>
    // CHECK: %[[CMP5:.+]] = arith.cmpf oge, %[[LOAD10]], %[[LOAD11]] : f32
    // CHECK: %[[EXT5:.+]] = arith.extui %[[CMP5]] : i1 to i8
    // CHECK: memref.store %[[EXT5]], %[[ALLOC4]][] : memref<i8>

    return x;
    // CHECK: return
}
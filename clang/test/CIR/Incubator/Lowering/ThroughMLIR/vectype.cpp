// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

typedef int vi4 __attribute__((vector_size(16)));

void vector_int_test(int x) {

  // CHECK: %[[ALLOC1:.*]] = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
  // CHECK: %[[ALLOC2:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC3:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC4:.*]] = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
  // CHECK: %[[ALLOC5:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC6:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC7:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC8:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC9:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC10:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC11:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC12:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC13:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC14:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC15:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC16:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC17:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC18:.*]] = memref.alloca() {alignment = 16 : i64} : memref<1xvector<4xi32>>

  // CHECK: memref.store %arg0, %[[ALLOC1]][{{%c0(_[0-9]+)?}}] : memref<1xi32>

  vi4 a = { 1, 2, 3, 4 };

  // CHECK: %[[CST:.*]] = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  // CHECK: memref.store %[[CST]], %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  
  vi4 b = {x, 5, 6, x + 1};

  // CHECK: %[[VAL1:.*]] = memref.load %[[ALLOC1]][{{%c0(_[0-9]+)?}}] : memref<1xi32>
  // CHECK: %[[C5:.*]] = arith.constant 5 : i32
  // CHECK: %[[C6:.*]] = arith.constant 6 : i32
  // CHECK: %[[VAL2:.*]] = memref.load %[[ALLOC1]][{{%c0(_[0-9]+)?}}] : memref<1xi32>
  // CHECK: %[[C1_I32_2:.*]] = arith.constant 1 : i32
  // CHECK: %[[SUM:.*]] = arith.addi %[[VAL2]], %[[C1_I32_2]] : i32
  // CHECK: %[[CST2:.*]] = arith.constant dense<0> : vector<4xi32>
  // CHECK: %[[VEC4:.*]] = vector.insert %[[VAL1]], %[[CST2]] [0] : i32 into vector<4xi32>
  // CHECK: %[[VEC5:.*]] = vector.insert %[[C5]], %[[VEC4]] [1] : i32 into vector<4xi32>
  // CHECK: %[[VEC6:.*]] = vector.insert %[[C6]], %[[VEC5]] [2] : i32 into vector<4xi32>
  // CHECK: %[[VEC7:.*]] = vector.insert %[[SUM]], %[[VEC6]] [3] : i32 into vector<4xi32>
  // CHECK: memref.store %[[VEC7]], %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  a[x] = x;
  
  // CHECK: %[[VAL3:.*]] = memref.load %[[ALLOC1]][{{%c0(_[0-9]+)?}}] : memref<1xi32>
  // CHECK: %[[VAL4:.*]] = memref.load %[[ALLOC1]][{{%c0(_[0-9]+)?}}] : memref<1xi32>
  // CHECK: %[[VEC8:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[IDX_CAST:.*]] = arith.index_cast %[[VAL4]] : i32 to index
  // CHECK: %[[VEC9:.*]] = vector.insert %[[VAL3]], %[[VEC8]] [%[[IDX_CAST]]] : i32 into vector<4xi32>
  // CHECK: memref.store %[[VEC9]], %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  int c = a[x];

  // CHECK: %[[VEC10:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[VAL5:.*]] = memref.load %[[ALLOC1]][{{%c0(_[0-9]+)?}}] : memref<1xi32>
  // CHECK: %[[IDX_CAST2:.*]] = arith.index_cast %[[VAL5]] : i32 to index
  // CHECK: %[[EXTRACT:.*]] = vector.extract %[[VEC10]][%[[IDX_CAST2]]] : i32 from vector<4xi32>
  // CHECK: memref.store %[[EXTRACT]], %[[ALLOC4]][{{%c0(_[0-9]+)?}}] : memref<1xi32>

  vi4 d = a + b;
  
  // CHECK: %[[ALLOC0_1:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC1_1:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC2_1:.*]] = arith.addi %[[ALLOC0_1]], %[[ALLOC1_1]] : vector<4xi32>
  // CHECK: memref.store %[[ALLOC2_1]], %[[ALLOC5]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  
  vi4 e = a - b;

  // CHECK: %[[ALLOC0_2:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC1_2:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC3_2:.*]] = arith.subi %[[ALLOC0_2]], %[[ALLOC1_2]] : vector<4xi32>
  // CHECK: memref.store %[[ALLOC3_2]], %[[ALLOC6]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 f = a * b;

  // CHECK: %[[ALLOC0_3:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC1_3:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC4_1:.*]] = arith.muli %[[ALLOC0_3]], %[[ALLOC1_3]] : vector<4xi32>
  // CHECK: memref.store %[[ALLOC4_1]], %[[ALLOC7]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 g = a / b;

  // CHECK: %[[ALLOC0_4:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC1_4:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC5_1:.*]] = arith.divsi %[[ALLOC0_4]], %[[ALLOC1_4]] : vector<4xi32>
  // CHECK: memref.store %[[ALLOC5_1]], %[[ALLOC8]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 h = a % b;

  // CHECK: %[[ALLOC0_5:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC1_5:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC6_1:.*]] = arith.remsi %[[ALLOC0_5]], %[[ALLOC1_5]] : vector<4xi32>
  // CHECK: memref.store %[[ALLOC6_1]], %[[ALLOC9]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 i = a & b;

  // CHECK: %[[ALLOC0_6:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC1_6:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC7_1:.*]] = arith.andi %[[ALLOC0_6]], %[[ALLOC1_6]] : vector<4xi32>
  // CHECK: memref.store %[[ALLOC7_1]], %[[ALLOC10]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 j = a | b;

  // CHECK: %[[ALLOC0_7:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC1_7:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC8_1:.*]] = arith.ori %[[ALLOC0_7]], %[[ALLOC1_7]] : vector<4xi32>
  // CHECK: memref.store %[[ALLOC8_1]], %[[ALLOC11]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 k = a ^ b;

  // CHECK: %[[ALLOC0_8:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC1_8:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[ALLOC9_1:.*]] = arith.xori %[[ALLOC0_8]], %[[ALLOC1_8]] : vector<4xi32>
  // CHECK: memref.store %[[ALLOC9_1]], %[[ALLOC12]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  // TODO(cir) : Fix the lowering of unary operators
  // vi4 l = +a;
  // vi4 m = -a;
  // vi4 n = ~a;

  vi4 o = a == b;

  // CHECK: %[[VAL11:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[VAL12:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[CMP_EQ:.*]] = arith.cmpi eq, %[[VAL11]], %[[VAL12]] : vector<4xi32>
  // CHECK: %[[EXT_EQ:.*]] = arith.extsi %[[CMP_EQ]] : vector<4xi1> to vector<4xi32>
  // CHECK: memref.store %[[EXT_EQ]], %[[ALLOC13]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 p = a != b;

  // CHECK: %[[VAL13:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[VAL14:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[CMP_NE:.*]] = arith.cmpi ne, %[[VAL13]], %[[VAL14]] : vector<4xi32>
  // CHECK: %[[EXT_NE:.*]] = arith.extsi %[[CMP_NE]] : vector<4xi1> to vector<4xi32>
  // CHECK: memref.store %[[EXT_NE]], %[[ALLOC14]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 q = a < b;

  // CHECK: %[[VAL15:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[VAL16:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[CMP_SLT:.*]] = arith.cmpi slt, %[[VAL15]], %[[VAL16]] : vector<4xi32>
  // CHECK: %[[EXT_SLT:.*]] = arith.extsi %[[CMP_SLT]] : vector<4xi1> to vector<4xi32>
  // CHECK: memref.store %[[EXT_SLT]], %[[ALLOC15]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  
  vi4 r = a > b;
  
  // CHECK: %[[VAL17:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[VAL18:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[CMP_SGT:.*]] = arith.cmpi sgt, %[[VAL17]], %[[VAL18]] : vector<4xi32>
  // CHECK: %[[EXT_SGT:.*]] = arith.extsi %[[CMP_SGT]] : vector<4xi1> to vector<4xi32>
  // CHECK: memref.store %[[EXT_SGT]], %[[ALLOC16]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 s = a <= b;

  // CHECK: %[[VAL19:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[VAL20:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[CMP_SLE:.*]] = arith.cmpi sle, %[[VAL19]], %[[VAL20]] : vector<4xi32>
  // CHECK: %[[EXT_SLE:.*]] = arith.extsi %[[CMP_SLE]] : vector<4xi1> to vector<4xi32>
  // CHECK: memref.store %[[EXT_SLE]], %[[ALLOC17]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  vi4 t = a >= b;

  // CHECK: %[[VAL21:.*]] = memref.load %[[ALLOC2]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[VAL22:.*]] = memref.load %[[ALLOC3]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>
  // CHECK: %[[CMP_SGE:.*]] = arith.cmpi sge, %[[VAL21]], %[[VAL22]] : vector<4xi32>
  // CHECK: %[[EXT_SGE:.*]] = arith.extsi %[[CMP_SGE]] : vector<4xi1> to vector<4xi32>
  // CHECK: memref.store %[[EXT_SGE]], %[[ALLOC18]][{{%c0(_[0-9]+)?}}] : memref<1xvector<4xi32>>

  // CHECK: return
}

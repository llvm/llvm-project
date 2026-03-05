// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void foo(int i) {}

int test(void) {
  foo(2);
  return 0;
}

// CHECK-LABEL: func.func @test() -> i32 {
//       CHECK:   %[[ARG:.+]] = arith.constant 2 : i32
//  CHECK-NEXT:   call @foo(%[[ARG]]) : (i32) -> ()
//       CHECK: }

extern int printf(const char *str, ...);

// CHECK-LABEL: llvm.func @printf(!llvm.ptr, ...) -> i32
//       CHECK: llvm.mlir.global internal constant @[[FRMT_STR:.*]](dense<[37, 100, 44, 32, 37, 102, 44, 32, 37, 100, 44, 32, 37, 108, 108, 100, 44, 32, 37, 100, 44, 32, 37, 102, 10, 0]> : tensor<26xi8>) {addr_space = 0 : i32} : !llvm.array<26 x i8>

void testfunc(short s, float X, char C, long long LL, int I, double D) {
	printf("%d, %f, %d, %lld, %d, %f\n", s, X, C, LL, I, D);
}

// CHECK: func.func @testfunc(%[[ARG0:.*]]: i16 {{.*}}, %[[ARG1:.*]]: f32 {{.*}}, %[[ARG2:.*]]: i8 {{.*}}, %[[ARG3:.*]]: i64 {{.*}}, %[[ARG4:.*]]: i32 {{.*}}, %[[ARG5:.*]]: f64 {{.*}}) {
// CHECK: %[[ALLOCA_S:.*]] = memref.alloca() {alignment = 2 : i64} : memref<1xi16>
// CHECK: %[[ALLOCA_X:.*]] = memref.alloca() {alignment = 4 : i64} : memref<1xf32>
// CHECK: %[[ALLOCA_C:.*]] = memref.alloca() {alignment = 1 : i64} : memref<1xi8>
// CHECK: %[[ALLOCA_LL:.*]] = memref.alloca() {alignment = 8 : i64} : memref<1xi64>
// CHECK: %[[ALLOCA_I:.*]] = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[ALLOCA_D:.*]] = memref.alloca() {alignment = 8 : i64} : memref<1xf64>
// CHECK: memref.store %[[ARG0]], %[[ALLOCA_S]][{{%c0(_[0-9]+)?}}] : memref<1xi16>
// CHECK: memref.store %[[ARG1]], %[[ALLOCA_X]][{{%c0(_[0-9]+)?}}] : memref<1xf32>
// CHECK: memref.store %[[ARG2]], %[[ALLOCA_C]][{{%c0(_[0-9]+)?}}] : memref<1xi8>
// CHECK: memref.store %[[ARG3]], %[[ALLOCA_LL]][{{%c0(_[0-9]+)?}}] : memref<1xi64>
// CHECK: memref.store %[[ARG4]], %[[ALLOCA_I]][{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: memref.store %[[ARG5]], %[[ALLOCA_D]][{{%c0(_[0-9]+)?}}] : memref<1xf64>
// CHECK: %[[FRMT_STR_ADDR:.*]] = llvm.mlir.addressof @[[FRMT_STR]] : !llvm.ptr 
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i8 
// CHECK: %[[FRMT_STR_DATA:.*]] = llvm.getelementptr %[[FRMT_STR_ADDR]][%[[C0]], %[[C0]]] : (!llvm.ptr, i8, i8) -> !llvm.ptr, !llvm.array<26 x i8>
// CHECK: %[[S:.*]] = memref.load %[[ALLOCA_S]][{{%c0(_[0-9]+)?}}] : memref<1xi16>
// CHECK: %[[S_EXT:.*]] = arith.extsi %3 : i16 to i32 
// CHECK: %[[X:.*]] = memref.load %[[ALLOCA_X]][{{%c0(_[0-9]+)?}}] : memref<1xf32>
// CHECK: %[[X_EXT:.*]] = arith.extf %5 : f32 to f64 
// CHECK: %[[C:.*]] = memref.load %[[ALLOCA_C]][{{%c0(_[0-9]+)?}}] : memref<1xi8>
// CHECK: %[[C_EXT:.*]] = arith.extsi %7 : i8 to i32
// CHECK: %[[LL:.*]] = memref.load %[[ALLOCA_LL]][{{%c0(_[0-9]+)?}}] : memref<1xi64>
// CHECK: %[[I:.*]] = memref.load %[[ALLOCA_I]][{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: %[[D:.*]] = memref.load %[[ALLOCA_D]][{{%c0(_[0-9]+)?}}] : memref<1xf64>
// CHECK: {{.*}} = llvm.call @printf(%[[FRMT_STR_DATA]], %[[S_EXT]], %[[X_EXT]], %[[C_EXT]], %[[LL]], %[[I]], %[[D]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, f64, i32, i64, i32, f64) -> i32 
// CHECK: return
// CHECK: } 

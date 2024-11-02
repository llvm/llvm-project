// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// check conversion of complex type of different size. Both fir and mlir
// variants are checked.

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "native"} {
  func.func @test1(%x : !fir.complex<4>) -> !fir.complex<8> {
  %1 = fir.convert %x : (!fir.complex<4>) -> !fir.complex<8>
  return %1 : !fir.complex<8>
  }loc(#loc1)
  func.func @test2(%x : !fir.complex<4>) -> complex<f64> {
  %1 = fir.convert %x : (!fir.complex<4>) -> complex<f64>
  return %1 : complex<f64>
  }loc(#loc2)
  func.func @test3(%x : !fir.complex<4>) -> !fir.complex<16> {
  %1 = fir.convert %x : (!fir.complex<4>) -> !fir.complex<16>
  return %1 : !fir.complex<16>
  }loc(#loc3)
  func.func @test4(%x : !fir.complex<4>) -> complex<f128> {
  %1 = fir.convert %x : (!fir.complex<4>) -> complex<f128>
  return %1 : complex<f128>
  }loc(#loc4)
}
#loc1 = loc("./simple.f90":2:1)
#loc2 = loc("./simple.f90":5:1)
#loc3 = loc("./simple.f90":8:1)
#loc4 = loc("./simple.f90":11:1)

// CHECK-DAG: #[[CMPX8:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex", sizeInBits = 128, encoding = DW_ATE_complex_float>
// CHECK-DAG: #[[CMPX4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex", sizeInBits = 64, encoding = DW_ATE_complex_float>
// CHECK-DAG: #[[CMPX16:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "complex", sizeInBits = 256, encoding = DW_ATE_complex_float>

// CHECK-DAG: #[[TY1:.*]] = #llvm.di_subroutine_type<{{.*}}types = #[[CMPX8]], #[[CMPX4]]>
// CHECK-DAG: #[[TY2:.*]] = #llvm.di_subroutine_type<{{.*}}types = #[[CMPX16]], #[[CMPX4]]>

// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "test1"{{.*}}type = #[[TY1]]>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "test2"{{.*}}type = #[[TY1]]>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "test3"{{.*}}type = #[[TY2]]>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}name = "test4"{{.*}}type = #[[TY2]]>

// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @test_codegen(%arg0: i32) {
    fir.declare_value %arg0 {uniq_name = "test"} : i32 loc(#loc1)
    return
  }
}

#di_file = #llvm.di_file<"test.f90" in "">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "Flang", isOptimized = false, emissionKind = Full>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "test_codegen", file = #di_file, subprogramFlags = Definition>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "test", file = #di_file, line = 1, arg = 1, alignInBits = 0, type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>>
#loc1 = loc(fused<#di_local_variable>["test.f90":1:1])

// CHECK-LABEL: llvm.func @test_codegen
// CHECK: llvm.intr.dbg.value

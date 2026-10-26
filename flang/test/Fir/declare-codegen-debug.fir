// Test that the debug intrinsic for a declared variable gets its location from
// the declaration and not from the allocation.
// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @test_codegen() {
    %0 = fir.alloca i32 loc(#loc_alloca)
    %1 = fircg.ext_declare %0 uniq_name("x") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc_x)
    %2 = fir.alloca i32 loc(unknown)
    %3 = fircg.ext_declare %2 uniq_name("y") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc_y)
    return
  }
}

#di_file = #llvm.di_file<"test.f90" in "">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "Flang", isOptimized = false, emissionKind = Full>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "test_codegen", file = #di_file, subprogramFlags = Definition>
#di_int = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer(kind=4)", sizeInBits = 32, encoding = DW_ATE_signed>
#di_x = #llvm.di_local_variable<scope = #di_subprogram, name = "x", file = #di_file, line = 3, type = #di_int>
#di_y = #llvm.di_local_variable<scope = #di_subprogram, name = "y", file = #di_file, line = 4, type = #di_int>
#loc_alloca = loc("test.f90":99:1)
#loc_x = loc(fused<#di_x>["test.f90":3:11])
#loc_y = loc(fused<#di_y>["test.f90":4:11])

// CHECK-LABEL: llvm.func @test_codegen
// CHECK:         llvm.intr.dbg.declare #[[X:[^ ]+]] = %{{[^ ]+}} : !llvm.ptr loc(#[[X_LOC:[^)]+]])
// CHECK:         llvm.intr.dbg.declare #[[Y:[^ ]+]] = %{{[^ ]+}} : !llvm.ptr loc(#[[Y_LOC:[^)]+]])
// CHECK:       #[[X_FILE:[^ ]+]] = loc("test.f90":3:11)
// CHECK:       #[[Y_FILE:[^ ]+]] = loc("test.f90":4:11)
// CHECK:       #[[X_LOC]] = loc(fused<#[[X]]>[#[[X_FILE]]])
// CHECK:       #[[Y_LOC]] = loc(fused<#[[Y]]>[#[[Y_FILE]]])

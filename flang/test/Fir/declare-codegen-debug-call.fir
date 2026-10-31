// Test the debug intrinsic for a variable whose storage address is returned by
// a call. Such a variable has no alloca to point at, so the declare must be
// given a home slot holding the address and dereference it.
// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func private @alloc() -> !fir.ref<!fir.array<100xf32>>
  func.func @test_address_from_call() {
    %0 = fir.call @alloc() : () -> !fir.ref<!fir.array<100xf32>>
    %1 = fircg.ext_declare %0 uniq_name("m") : (!fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>> loc(#loc_m)
    return
  }

  func.func @test_address_from_allocmem() {
    %0 = fir.allocmem !fir.array<100xf32>
    %1 = fircg.ext_declare %0 uniq_name("h") : (!fir.heap<!fir.array<100xf32>>) -> !fir.heap<!fir.array<100xf32>> loc(#loc_h)
    fir.freemem %1 : !fir.heap<!fir.array<100xf32>>
    return
  }
}

#di_file = #llvm.di_file<"test.cuf" in "">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "Flang", isOptimized = false, emissionKind = Full>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "test_address_from_call", file = #di_file, subprogramFlags = Definition>
#di_subprogram_heap = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "test_address_from_allocmem", file = #di_file, subprogramFlags = Definition>
#di_real = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=4)", sizeInBits = 32, encoding = DW_ATE_float>
#di_m = #llvm.di_local_variable<scope = #di_subprogram, name = "m", file = #di_file, line = 4, type = #di_real>
#di_h = #llvm.di_local_variable<scope = #di_subprogram_heap, name = "h", file = #di_file, line = 9, type = #di_real>
#loc_m = loc(fused<#di_m>["test.cuf":4:11])
#loc_h = loc(fused<#di_h>["test.cuf":9:11])

// CHECK-LABEL: llvm.func @test_address_from_call
// CHECK:         %[[SLOT:[^ ]+]] = llvm.alloca %{{[^ ]+}} x !llvm.ptr
// CHECK:         %[[ADDR:[^ ]+]] = llvm.call @alloc()
// CHECK:         llvm.store %[[ADDR]], %[[SLOT]]
// CHECK:         llvm.intr.dbg.declare #[[M:[^ ]+]] #llvm.di_expression<[DW_OP_deref]> = %[[SLOT]] : !llvm.ptr

// CHECK-LABEL: llvm.func @test_address_from_allocmem
// CHECK:         %[[HSLOT:[^ ]+]] = llvm.alloca %{{[^ ]+}} x !llvm.ptr
// CHECK:         %[[HADDR:[^ ]+]] = llvm.call @malloc
// CHECK:         llvm.store %[[HADDR]], %[[HSLOT]]
// CHECK:         llvm.intr.dbg.declare #[[H:[^ ]+]] #llvm.di_expression<[DW_OP_deref]> = %[[HSLOT]] : !llvm.ptr

// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  fir.global @_QMmEvar : !fir.type<_QMmTt1{elm:!fir.array<5xi32>,elm2:!fir.array<5x8xi32>}> {} loc(#loc1)
  fir.type_info @_QMmTt1 noinit nodestroy nofinal : !fir.type<_QMmTt1{elm:!fir.array<5xi32>,elm2:!fir.array<5x8xi32>}> component_info {
    fir.dt_component "elm" lbs [2]
    fir.dt_component "elm2" lbs [1, 3]
  } loc(#loc1)
}
#loc1 = loc("derived.f90":24:1)


// CHECK-DAG: #[[TY1:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}} = #{{.*}}, elements = #llvm.di_subrange<count = 5 : i64, lowerBound = 2 : i64>>
// CHECK-DAG: #[[TY2:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #{{.*}}, elements = #llvm.di_subrange<count = 5 : i64, lowerBound = 1 : i64>, #llvm.di_subrange<count = 8 : i64, lowerBound = 3 : i64>>
// CHECK-DAG: #llvm.di_derived_type<tag = DW_TAG_member, name = "elm", baseType = #[[TY1]], sizeInBits = {{.*}}, alignInBits = {{.*}}>
// CHECK-DAG: #llvm.di_derived_type<tag = DW_TAG_member, name = "elm2", baseType = #[[TY2]], sizeInBits = {{.*}}, alignInBits = {{.*}}>

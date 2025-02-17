// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  fir.type_info @_QTtest_type nofinal : !fir.type<_QTtest_type{a:i32,b:!fir.box<!fir.heap<!fir.array<?x?xf32>>>}> dispatch_table {
    fir.dt_entry "test_proc", @_QPtest_proc
  } loc(#loc1)
  func.func private @_QPtest_proc(%arg0: !fir.class<!fir.type<_QTtest_type{a:i32,b:!fir.box<!fir.heap<!fir.array<?x?xf32>>>}>>)

    func.func @test()  {
    %0 = fir.address_of(@_QFEx) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QTtest_type{a:i32,b:!fir.box<!fir.heap<!fir.array<?x?xf32>>>}>>>>
    %1 = fircg.ext_declare %0 {uniq_name = "_QFEx"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QTtest_type{a:i32,b:!fir.box<!fir.heap<!fir.array<?x?xf32>>>}>>>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QTtest_type{a:i32,b:!fir.box<!fir.heap<!fir.array<?x?xf32>>>}>>>> loc(#loc3)
    %2 = fir.address_of(@_QFEy) : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QTtest_type{a:i32,b:!fir.box<!fir.heap<!fir.array<?x?xf32>>>}>>>>
    %3 = fircg.ext_declare %2 {uniq_name = "_QFEy"} : (!fir.ref<!fir.class<!fir.ptr<!fir.type<_QTtest_type{a:i32,b:!fir.box<!fir.heap<!fir.array<?x?xf32>>>}>>>>) -> !fir.ref<!fir.class<!fir.ptr<!fir.type<_QTtest_type{a:i32,b:!fir.box<!fir.heap<!fir.array<?x?xf32>>>}>>>> loc(#loc4)
    %4 = fir.address_of(@_QFEz) : !fir.ref<!fir.class<none>>
    %5 = fircg.ext_declare %4 {uniq_name = "_QFEz"} : (!fir.ref<!fir.class<none>>) -> !fir.ref<!fir.class<none>> loc(#loc4)
    %6 = fir.address_of(@_QFEt) : !fir.ref<!fir.class<!fir.ptr<none>>>
    %7 = fircg.ext_declare %6 {uniq_name = "_QFEt"} : (!fir.ref<!fir.class<!fir.ptr<none>>>) -> !fir.ref<!fir.class<!fir.ptr<none>>> loc(#loc4)
    return
  } loc(#loc2)
}

#loc1 = loc("./simple.f90":2:1)
#loc2 = loc("./simple.f90":10:1)
#loc3 = loc("./simple.f90":15:1)
#loc4 = loc("./simple.f90":22:1)

// CHECK-DAG: #[[TY1:.*]] = #llvm.di_composite_type<{{.*}}name = "test_type"{{.*}}>
// CHECK-DAG: #[[TY2:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, name = "", baseType = #[[TY1]]{{.*}}>
// CHECK-DAG: #[[TY3:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, name = "", baseType = #di_null_type{{.*}}>
// CHECK-DAG: #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_null_type, #[[TY2]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "x"{{.*}}type = #[[TY2]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "y"{{.*}}type = #[[TY2]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "z"{{.*}}type = #[[TY3]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "t"{{.*}}type = #[[TY3]]>

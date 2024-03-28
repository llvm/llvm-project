// Test fir alias analysis pass on a larger real life code example (from the RFC)
// RUN: fir-opt --fir-add-alias-tags %s | FileCheck %s

  fir.global @_QMmodEa : !fir.box<!fir.heap<!fir.array<?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
  }
  fir.global @_QMmodEb : !fir.box<!fir.heap<!fir.array<?xf32>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
  }
  fir.global @_QMmodEdxinv : f32 {
    %0 = fir.zero_bits f32
    fir.has_value %0 : f32
  }
  fir.global @_QMmodEdyinv : f32 {
    %0 = fir.zero_bits f32
    fir.has_value %0 : f32
  }
  fir.global @_QMmodExstart : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  fir.global @_QMmodEystart : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  fir.global @_QMmodEystop : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  fir.global @_QMmodEzstart : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  fir.global @_QMmodEzstop : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }

// CHECK: #[[ROOT:.+]] = #ptr.tbaa_root<id = "Flang function root _QMmodPcallee">
// CHECK: #[[ANY_ACCESS:.+]] = #ptr.tbaa_type_desc<id = "any access", members = {<#[[ROOT]], 0>}>
// CHECK: #[[ANY_DATA:.+]] = #ptr.tbaa_type_desc<id = "any data access", members = {<#[[ANY_ACCESS]], 0>}>
// CHECK: #[[ANY_GLBL:.+]] = #ptr.tbaa_type_desc<id = "global data", members = {<#[[ANY_DATA]], 0>}>
// CHECK: #[[ANY_ARG:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data", members = {<#[[ANY_DATA]], 0>}>
// CHECK: #[[ANY_DIRECT:.+]] = #ptr.tbaa_type_desc<id = "direct data", members = {<#[[ANY_DATA]], 0>}>
// CHECK: #[[GLBL_ZSTART:.+]] = #ptr.tbaa_type_desc<id = "global data/_QMmodEzstart", members = {<#[[ANY_GLBL]], 0>}>
// CHECK: #[[GLBL_ZSTOP:.+]] = #ptr.tbaa_type_desc<id = "global data/_QMmodEzstop", members = {<#[[ANY_GLBL]], 0>}>
// CHECK: #[[GLBL_YSTART:.+]] = #ptr.tbaa_type_desc<id = "global data/_QMmodEystart", members = {<#[[ANY_GLBL]], 0>}>
// CHECK: #[[GLBL_YSTOP:.+]] = #ptr.tbaa_type_desc<id = "global data/_QMmodEystop", members = {<#[[ANY_GLBL]], 0>}>
// CHECK: #[[GLBL_XSTART:.+]] = #ptr.tbaa_type_desc<id = "global data/_QMmodExstart", members = {<#[[ANY_GLBL]], 0>}>
// CHECK: #[[ARG_LOW:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data/_QMmodFcalleeElow", members = {<#[[ANY_ARG]], 0>}>
// CHECK: #[[DIRECT_A:.+]] = #ptr.tbaa_type_desc<id = "direct data/_QMmodEa", members = {<#[[ANY_DIRECT]], 0>}>
// CHECK: #[[DIRECT_B:.+]] = #ptr.tbaa_type_desc<id = "direct data/_QMmodEb", members = {<#[[ANY_DIRECT]], 0>}>
// CHECK: #[[ARG_Z:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data/_QMmodFcalleeEz", members = {<#[[ANY_ARG]], 0>}>
// CHECK: #[[GLBL_DYINV:.+]] = #ptr.tbaa_type_desc<id = "global data/_QMmodEdyinv", members = {<#[[ANY_GLBL]], 0>}>
// CHECK: #[[ARG_Y:.+]] = #ptr.tbaa_type_desc<id = "dummy arg data/_QMmodFcalleeEy", members = {<#[[ANY_ARG]], 0>}>

// CHECK: #[[GLBL_ZSTART_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[GLBL_ZSTART]], access_type = #[[GLBL_ZSTART]], offset = 0>
// CHECK: #[[GLBL_ZSTOP_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[GLBL_ZSTOP]], access_type = #[[GLBL_ZSTOP]], offset = 0>
// CHECK: #[[GLBL_YSTART_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[GLBL_YSTART]], access_type = #[[GLBL_YSTART]], offset = 0>
// CHECK: #[[GLBL_YSTOP_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[GLBL_YSTOP]], access_type = #[[GLBL_YSTOP]], offset = 0>
// CHECK: #[[GLBL_XSTART_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[GLBL_XSTART]], access_type = #[[GLBL_XSTART]], offset = 0>
// CHECK: #[[ARG_LOW_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[ARG_LOW]], access_type = #[[ARG_LOW]], offset = 0>
// CHECK: #[[DIRECT_A_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[DIRECT_A]], access_type = #[[DIRECT_A]], offset = 0>
// CHECK: #[[DIRECT_B_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[DIRECT_B]], access_type = #[[DIRECT_B]], offset = 0>
// CHECK: #[[ARG_Z_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[ARG_Z]], access_type = #[[ARG_Z]], offset = 0>
// CHECK: #[[GLBL_DYINV_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[GLBL_DYINV]], access_type = #[[GLBL_DYINV]], offset = 0>
// CHECK: #[[ARG_Y_TAG:.+]] = #ptr.tbaa_tag<base_type = #[[ARG_Y]], access_type = #[[ARG_Y]], offset = 0>

  func.func @_QMmodPcallee(%arg0: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "z"}, %arg1: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "y"}, %arg2: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>> {fir.bindc_name = "low"}) {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = fir.address_of(@_QMmodEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmodEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %2 = fir.address_of(@_QMmodEb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %3 = fir.declare %2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmodEb"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %4 = fir.address_of(@_QMmodEdxinv) : !fir.ref<f32>
    %5 = fir.declare %4 {uniq_name = "_QMmodEdxinv"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %6 = fir.address_of(@_QMmodEdyinv) : !fir.ref<f32>
    %7 = fir.declare %6 {uniq_name = "_QMmodEdyinv"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %8 = fir.address_of(@_QMmodExstart) : !fir.ref<i32>
    %9 = fir.declare %8 {uniq_name = "_QMmodExstart"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %10 = fir.address_of(@_QMmodEystart) : !fir.ref<i32>
    %11 = fir.declare %10 {uniq_name = "_QMmodEystart"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %12 = fir.address_of(@_QMmodEystop) : !fir.ref<i32>
    %13 = fir.declare %12 {uniq_name = "_QMmodEystop"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %14 = fir.address_of(@_QMmodEzstart) : !fir.ref<i32>
    %15 = fir.declare %14 {uniq_name = "_QMmodEzstart"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %16 = fir.address_of(@_QMmodEzstop) : !fir.ref<i32>
    %17 = fir.declare %16 {uniq_name = "_QMmodEzstop"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %18 = fir.alloca f32 {bindc_name = "dxold", uniq_name = "_QMmodFcalleeEdxold"}
    %19 = fir.declare %18 {uniq_name = "_QMmodFcalleeEdxold"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %20 = fir.alloca f32 {bindc_name = "dzinv", uniq_name = "_QMmodFcalleeEdzinv"}
    %21 = fir.declare %20 {uniq_name = "_QMmodFcalleeEdzinv"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %22 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMmodFcalleeEi"}
    %23 = fir.declare %22 {uniq_name = "_QMmodFcalleeEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %24 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QMmodFcalleeEj"}
    %25 = fir.declare %24 {uniq_name = "_QMmodFcalleeEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %26 = fir.alloca i32 {bindc_name = "k", uniq_name = "_QMmodFcalleeEk"}
    %27 = fir.declare %26 {uniq_name = "_QMmodFcalleeEk"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %28 = fir.declare %arg2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmodFcalleeElow"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>
    %29 = fir.declare %arg1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMmodFcalleeEy"} : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
    %30 = fir.rebox %29 : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
    %31 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMmodFcalleeEz"} : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
    %32 = fir.rebox %31 : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
    %33 = fir.load %15 : !fir.ref<i32>
    %34 = arith.addi %33, %c1_i32 : i32
    %35 = fir.convert %34 : (i32) -> index
    %36 = fir.load %17 : !fir.ref<i32>
    %37 = fir.convert %36 : (i32) -> index
    %38 = fir.convert %35 : (index) -> i32
    %39:2 = fir.do_loop %arg3 = %35 to %37 step %c1 iter_args(%arg4 = %38) -> (index, i32) {
      fir.store %arg4 to %27 : !fir.ref<i32>
      %40 = fir.load %11 : !fir.ref<i32>
      %41 = arith.addi %40, %c1_i32 : i32
      %42 = fir.convert %41 : (i32) -> index
      %43 = fir.load %13 : !fir.ref<i32>
      %44 = fir.convert %43 : (i32) -> index
      %45 = fir.convert %42 : (index) -> i32
      %46:2 = fir.do_loop %arg5 = %42 to %44 step %c1 iter_args(%arg6 = %45) -> (index, i32) {
        fir.store %arg6 to %25 : !fir.ref<i32>
        %51 = fir.load %9 : !fir.ref<i32>
        %52 = arith.addi %51, %c1_i32 : i32
        %53 = fir.convert %52 : (i32) -> index
        %54 = fir.convert %53 : (index) -> i32
        %55:2 = fir.do_loop %arg7 = %53 to %c0 step %c1 iter_args(%arg8 = %54) -> (index, i32) {
          fir.store %arg8 to %23 : !fir.ref<i32>
          %60 = fir.load %28 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>
          %61 = fir.load %23 : !fir.ref<i32>
          %62 = fir.convert %61 : (i32) -> i64
          %63 = fir.load %25 : !fir.ref<i32>
          %64 = fir.convert %63 : (i32) -> i64
          %65 = fir.load %27 : !fir.ref<i32>
          %66 = fir.convert %65 : (i32) -> i64
          %67 = fir.box_addr %60 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>) -> !fir.heap<!fir.array<?x?x?xf32>>
          %68:3 = fir.box_dims %60, %c0 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
          %69:3 = fir.box_dims %60, %c1 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
          %70:3 = fir.box_dims %60, %c2 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
          %71 = fir.shape_shift %68#0, %68#1, %69#0, %69#1, %70#0, %70#1 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
          %72 = fir.array_coor %67(%71) %62, %64, %66 : (!fir.heap<!fir.array<?x?x?xf32>>, !fir.shapeshift<3>, i64, i64, i64) -> !fir.ref<f32>
          %73 = fir.load %72 : !fir.ref<f32>
          fir.store %73 to %19 : !fir.ref<f32>
          %74 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
          %75 = fir.load %25 : !fir.ref<i32>
          %76 = fir.convert %75 : (i32) -> i64
          %77 = fir.box_addr %74 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
          %78:3 = fir.box_dims %74, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
          %79 = fir.shape_shift %78#0, %78#1 : (index, index) -> !fir.shapeshift<1>
          %80 = fir.array_coor %77(%79) %76 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, i64) -> !fir.ref<f32>
          %81 = fir.load %80 : !fir.ref<f32>
          %82 = fir.load %28 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>
          %83 = fir.load %23 : !fir.ref<i32>
          %84 = fir.convert %83 : (i32) -> i64
          %85 = fir.load %27 : !fir.ref<i32>
          %86 = fir.convert %85 : (i32) -> i64
          %87 = fir.box_addr %82 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>) -> !fir.heap<!fir.array<?x?x?xf32>>
          %88:3 = fir.box_dims %82, %c0 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
          %89:3 = fir.box_dims %82, %c1 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
          %90:3 = fir.box_dims %82, %c2 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
          %91 = fir.shape_shift %88#0, %88#1, %89#0, %89#1, %90#0, %90#1 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
          %92 = fir.array_coor %87(%91) %84, %76, %86 : (!fir.heap<!fir.array<?x?x?xf32>>, !fir.shapeshift<3>, i64, i64, i64) -> !fir.ref<f32>
          %93 = fir.load %92 : !fir.ref<f32>
          %94 = arith.mulf %81, %93 fastmath<contract> : f32
          %95 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
          %96 = fir.box_addr %95 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
          %97:3 = fir.box_dims %95, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
          %98 = fir.shape_shift %97#0, %97#1 : (index, index) -> !fir.shapeshift<1>
          %99 = fir.array_coor %96(%98) %76 : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, i64) -> !fir.ref<f32>
          %100 = fir.load %99 : !fir.ref<f32>
          %101 = fir.array_coor %32 %84, %76, %86 : (!fir.box<!fir.array<?x?x?xf32>>, i64, i64, i64) -> !fir.ref<f32>
          %102 = fir.load %101 : !fir.ref<f32>
          %103 = arith.subi %75, %c1_i32 : i32
          %104 = fir.convert %103 : (i32) -> i64
          %105 = fir.array_coor %32 %84, %104, %86 : (!fir.box<!fir.array<?x?x?xf32>>, i64, i64, i64) -> !fir.ref<f32>
          %106 = fir.load %105 : !fir.ref<f32>
          %107 = arith.subf %102, %106 fastmath<contract> : f32
          %108 = fir.no_reassoc %107 : f32
          %109 = fir.load %7 : !fir.ref<f32>
          %110 = arith.mulf %108, %109 fastmath<contract> : f32
          %111 = arith.subi %85, %c1_i32 : i32
          %112 = fir.convert %111 : (i32) -> i64
          %113 = fir.array_coor %30 %84, %76, %112 : (!fir.box<!fir.array<?x?x?xf32>>, i64, i64, i64) -> !fir.ref<f32>
          %114 = fir.load %113 : !fir.ref<f32>
          %115 = fir.array_coor %30 %84, %76, %86 : (!fir.box<!fir.array<?x?x?xf32>>, i64, i64, i64) -> !fir.ref<f32>
          %116 = fir.load %115 : !fir.ref<f32>
          %117 = arith.subf %114, %116 fastmath<contract> : f32
          %118 = fir.no_reassoc %117 : f32
          %119 = fir.load %21 : !fir.ref<f32>
          %120 = arith.mulf %118, %119 fastmath<contract> : f32
          %121 = arith.addf %110, %120 fastmath<contract> : f32
          %122 = fir.no_reassoc %121 : f32
          %123 = arith.mulf %100, %122 fastmath<contract> : f32
          %124 = arith.addf %94, %123 fastmath<contract> : f32
          fir.store %124 to %92 : !fir.ref<f32>
          %125 = arith.addi %arg7, %c1 : index
          %126 = fir.convert %c1 : (index) -> i32
          %127 = fir.load %23 : !fir.ref<i32>
          %128 = arith.addi %127, %126 : i32
          fir.result %125, %128 : index, i32
        }
        fir.store %55#1 to %23 : !fir.ref<i32>
        %56 = arith.addi %arg5, %c1 : index
        %57 = fir.convert %c1 : (index) -> i32
        %58 = fir.load %25 : !fir.ref<i32>
        %59 = arith.addi %58, %57 : i32
        fir.result %56, %59 : index, i32
      }
      fir.store %46#1 to %25 : !fir.ref<i32>
      %47 = arith.addi %arg3, %c1 : index
      %48 = fir.convert %c1 : (index) -> i32
      %49 = fir.load %27 : !fir.ref<i32>
      %50 = arith.addi %49, %48 : i32
      fir.result %47, %50 : index, i32
    }
    fir.store %39#1 to %27 : !fir.ref<i32>
    return
  }
// CHECK-LABEL:   func.func @_QMmodPcallee(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "z"},
// CHECK-SAME:                             %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "y"},
// CHECK-SAME:                             %[[VAL_2:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>> {fir.bindc_name = "low"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QMmodEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_7]] {fortran_attrs = #{{.*}}<allocatable>, uniq_name = "_QMmodEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QMmodEb) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_10:.*]] = fir.declare %[[VAL_9]] {fortran_attrs = #{{.*}}<allocatable>, uniq_name = "_QMmodEb"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QMmodEdxinv) : !fir.ref<f32>
// CHECK:           %[[VAL_12:.*]] = fir.declare %[[VAL_11]] {uniq_name = "_QMmodEdxinv"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QMmodEdyinv) : !fir.ref<f32>
// CHECK:           %[[VAL_14:.*]] = fir.declare %[[VAL_13]] {uniq_name = "_QMmodEdyinv"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_15:.*]] = fir.address_of(@_QMmodExstart) : !fir.ref<i32>
// CHECK:           %[[VAL_16:.*]] = fir.declare %[[VAL_15]] {uniq_name = "_QMmodExstart"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_17:.*]] = fir.address_of(@_QMmodEystart) : !fir.ref<i32>
// CHECK:           %[[VAL_18:.*]] = fir.declare %[[VAL_17]] {uniq_name = "_QMmodEystart"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_19:.*]] = fir.address_of(@_QMmodEystop) : !fir.ref<i32>
// CHECK:           %[[VAL_20:.*]] = fir.declare %[[VAL_19]] {uniq_name = "_QMmodEystop"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_21:.*]] = fir.address_of(@_QMmodEzstart) : !fir.ref<i32>
// CHECK:           %[[VAL_22:.*]] = fir.declare %[[VAL_21]] {uniq_name = "_QMmodEzstart"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_23:.*]] = fir.address_of(@_QMmodEzstop) : !fir.ref<i32>
// CHECK:           %[[VAL_24:.*]] = fir.declare %[[VAL_23]] {uniq_name = "_QMmodEzstop"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_25:.*]] = fir.alloca f32 {bindc_name = "dxold", uniq_name = "_QMmodFcalleeEdxold"}
// CHECK:           %[[VAL_26:.*]] = fir.declare %[[VAL_25]] {uniq_name = "_QMmodFcalleeEdxold"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_27:.*]] = fir.alloca f32 {bindc_name = "dzinv", uniq_name = "_QMmodFcalleeEdzinv"}
// CHECK:           %[[VAL_28:.*]] = fir.declare %[[VAL_27]] {uniq_name = "_QMmodFcalleeEdzinv"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_29:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMmodFcalleeEi"}
// CHECK:           %[[VAL_30:.*]] = fir.declare %[[VAL_29]] {uniq_name = "_QMmodFcalleeEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_31:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QMmodFcalleeEj"}
// CHECK:           %[[VAL_32:.*]] = fir.declare %[[VAL_31]] {uniq_name = "_QMmodFcalleeEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_33:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QMmodFcalleeEk"}
// CHECK:           %[[VAL_34:.*]] = fir.declare %[[VAL_33]] {uniq_name = "_QMmodFcalleeEk"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_35:.*]] = fir.declare %[[VAL_2]] {fortran_attrs = #{{.*}}<allocatable>, uniq_name = "_QMmodFcalleeElow"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>
// CHECK:           %[[VAL_36:.*]] = fir.declare %[[VAL_1]] {fortran_attrs = #{{.*}}<intent_in>, uniq_name = "_QMmodFcalleeEy"} : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK:           %[[VAL_37:.*]] = fir.rebox %[[VAL_36]] : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK:           %[[VAL_38:.*]] = fir.declare %[[VAL_0]] {fortran_attrs = #{{.*}}<intent_in>, uniq_name = "_QMmodFcalleeEz"} : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK:           %[[VAL_39:.*]] = fir.rebox %[[VAL_38]] : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_22]] {tbaa = [#[[GLBL_ZSTART_TAG]]]} : !fir.ref<i32>
// CHECK:           %[[VAL_41:.*]] = arith.addi %[[VAL_40]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> index
// CHECK:           %[[VAL_43:.*]] = fir.load %[[VAL_24]] {tbaa = [#[[GLBL_ZSTOP_TAG]]]} : !fir.ref<i32>
// CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> index
// CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_42]] : (index) -> i32
// CHECK:           %[[VAL_46:.*]]:2 = fir.do_loop %[[VAL_47:.*]] = %[[VAL_42]] to %[[VAL_44]] step %[[VAL_5]] iter_args(%[[VAL_48:.*]] = %[[VAL_45]]) -> (index, i32) {
// CHECK:             fir.store %[[VAL_48]] to %[[VAL_34]] : !fir.ref<i32>
// CHECK:             %[[VAL_49:.*]] = fir.load %[[VAL_18]] {tbaa = [#[[GLBL_YSTART_TAG]]]} : !fir.ref<i32>
// CHECK:             %[[VAL_50:.*]] = arith.addi %[[VAL_49]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i32) -> index
// CHECK:             %[[VAL_52:.*]] = fir.load %[[VAL_20]] {tbaa = [#[[GLBL_YSTOP_TAG]]]} : !fir.ref<i32>
// CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i32) -> index
// CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_51]] : (index) -> i32
// CHECK:             %[[VAL_55:.*]]:2 = fir.do_loop %[[VAL_56:.*]] = %[[VAL_51]] to %[[VAL_53]] step %[[VAL_5]] iter_args(%[[VAL_57:.*]] = %[[VAL_54]]) -> (index, i32) {
// CHECK:               fir.store %[[VAL_57]] to %[[VAL_32]] : !fir.ref<i32>
// CHECK:               %[[VAL_58:.*]] = fir.load %[[VAL_16]] {tbaa = [#[[GLBL_XSTART_TAG]]]} : !fir.ref<i32>
// CHECK:               %[[VAL_59:.*]] = arith.addi %[[VAL_58]], %[[VAL_6]] : i32
// CHECK:               %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i32) -> index
// CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (index) -> i32
// CHECK:               %[[VAL_62:.*]]:2 = fir.do_loop %[[VAL_63:.*]] = %[[VAL_60]] to %[[VAL_4]] step %[[VAL_5]] iter_args(%[[VAL_64:.*]] = %[[VAL_61]]) -> (index, i32) {
// TODO: local allocation assumed to always alias
// CHECK:                 fir.store %[[VAL_64]] to %[[VAL_30]] : !fir.ref<i32>
// load from box tagged in CodeGen
// CHECK:                 %[[VAL_65:.*]] = fir.load %[[VAL_35]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>
// TODO: local allocation assumed to always alias
// CHECK:                 %[[VAL_66:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
// CHECK:                 %[[VAL_67:.*]] = fir.convert %[[VAL_66]] : (i32) -> i64
// TODO: local allocation assumed to always alias
// CHECK:                 %[[VAL_68:.*]] = fir.load %[[VAL_32]] : !fir.ref<i32>
// CHECK:                 %[[VAL_69:.*]] = fir.convert %[[VAL_68]] : (i32) -> i64
// TODO: local allocation assumed to always alias
// CHECK:                 %[[VAL_70:.*]] = fir.load %[[VAL_34]] : !fir.ref<i32>
// CHECK:                 %[[VAL_71:.*]] = fir.convert %[[VAL_70]] : (i32) -> i64
// CHECK:                 %[[VAL_72:.*]] = fir.box_addr %[[VAL_65]] : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>) -> !fir.heap<!fir.array<?x?x?xf32>>
// CHECK:                 %[[VAL_73:.*]]:3 = fir.box_dims %[[VAL_65]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_74:.*]]:3 = fir.box_dims %[[VAL_65]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_75:.*]]:3 = fir.box_dims %[[VAL_65]], %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_76:.*]] = fir.shape_shift %[[VAL_73]]#0, %[[VAL_73]]#1, %[[VAL_74]]#0, %[[VAL_74]]#1, %[[VAL_75]]#0, %[[VAL_75]]#1 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
// CHECK:                 %[[VAL_77:.*]] = fir.array_coor %[[VAL_72]](%[[VAL_76]]) %[[VAL_67]], %[[VAL_69]], %[[VAL_71]] : (!fir.heap<!fir.array<?x?x?xf32>>, !fir.shapeshift<3>, i64, i64, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_78:.*]] = fir.load %[[VAL_77]] {tbaa = [#[[ARG_LOW_TAG]]]} : !fir.ref<f32>
// CHECK:                 fir.store %[[VAL_78]] to %[[VAL_26]] : !fir.ref<f32>
// load from box tagged in CodeGen
// CHECK:                 %[[VAL_79:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// TODO: local allocation assumed to always alias
// CHECK:                 %[[VAL_80:.*]] = fir.load %[[VAL_32]] : !fir.ref<i32>
// CHECK:                 %[[VAL_81:.*]] = fir.convert %[[VAL_80]] : (i32) -> i64
// CHECK:                 %[[VAL_82:.*]] = fir.box_addr %[[VAL_79]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:                 %[[VAL_83:.*]]:3 = fir.box_dims %[[VAL_79]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_84:.*]] = fir.shape_shift %[[VAL_83]]#0, %[[VAL_83]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:                 %[[VAL_85:.*]] = fir.array_coor %[[VAL_82]](%[[VAL_84]]) %[[VAL_81]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_86:.*]] = fir.load %[[VAL_85]] {tbaa = [#[[DIRECT_A_TAG]]]} : !fir.ref<f32>
// load from box
// CHECK:                 %[[VAL_87:.*]] = fir.load %[[VAL_35]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>
// load from local allocation
// CHECK:                 %[[VAL_88:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
// CHECK:                 %[[VAL_89:.*]] = fir.convert %[[VAL_88]] : (i32) -> i64
// load from local allocation
// CHECK:                 %[[VAL_90:.*]] = fir.load %[[VAL_34]] : !fir.ref<i32>
// CHECK:                 %[[VAL_91:.*]] = fir.convert %[[VAL_90]] : (i32) -> i64
// CHECK:                 %[[VAL_92:.*]] = fir.box_addr %[[VAL_87]] : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>) -> !fir.heap<!fir.array<?x?x?xf32>>
// CHECK:                 %[[VAL_93:.*]]:3 = fir.box_dims %[[VAL_87]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_94:.*]]:3 = fir.box_dims %[[VAL_87]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_95:.*]]:3 = fir.box_dims %[[VAL_87]], %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_96:.*]] = fir.shape_shift %[[VAL_93]]#0, %[[VAL_93]]#1, %[[VAL_94]]#0, %[[VAL_94]]#1, %[[VAL_95]]#0, %[[VAL_95]]#1 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
// CHECK:                 %[[VAL_97:.*]] = fir.array_coor %[[VAL_92]](%[[VAL_96]]) %[[VAL_89]], %[[VAL_81]], %[[VAL_91]] : (!fir.heap<!fir.array<?x?x?xf32>>, !fir.shapeshift<3>, i64, i64, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_98:.*]] = fir.load %[[VAL_97]] {tbaa = [#[[ARG_LOW_TAG]]]} : !fir.ref<f32>
// CHECK:                 %[[VAL_99:.*]] = arith.mulf %[[VAL_86]], %[[VAL_98]] fastmath<contract> : f32
// load from box
// CHECK:                 %[[VAL_100:.*]] = fir.load %[[VAL_10]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:                 %[[VAL_101:.*]] = fir.box_addr %[[VAL_100]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:                 %[[VAL_102:.*]]:3 = fir.box_dims %[[VAL_100]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_103:.*]] = fir.shape_shift %[[VAL_102]]#0, %[[VAL_102]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:                 %[[VAL_104:.*]] = fir.array_coor %[[VAL_101]](%[[VAL_103]]) %[[VAL_81]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_105:.*]] = fir.load %[[VAL_104]] {tbaa = [#[[DIRECT_B_TAG]]]} : !fir.ref<f32>
// CHECK:                 %[[VAL_106:.*]] = fir.array_coor %[[VAL_39]] %[[VAL_89]], %[[VAL_81]], %[[VAL_91]] : (!fir.box<!fir.array<?x?x?xf32>>, i64, i64, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_107:.*]] = fir.load %[[VAL_106]] {tbaa = [#[[ARG_Z_TAG]]]} : !fir.ref<f32>
// CHECK:                 %[[VAL_108:.*]] = arith.subi %[[VAL_80]], %[[VAL_6]] : i32
// CHECK:                 %[[VAL_109:.*]] = fir.convert %[[VAL_108]] : (i32) -> i64
// CHECK:                 %[[VAL_110:.*]] = fir.array_coor %[[VAL_39]] %[[VAL_89]], %[[VAL_109]], %[[VAL_91]] : (!fir.box<!fir.array<?x?x?xf32>>, i64, i64, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_111:.*]] = fir.load %[[VAL_110]] {tbaa = [#[[ARG_Z_TAG]]]} : !fir.ref<f32>
// CHECK:                 %[[VAL_112:.*]] = arith.subf %[[VAL_107]], %[[VAL_111]] fastmath<contract> : f32
// CHECK:                 %[[VAL_113:.*]] = fir.no_reassoc %[[VAL_112]] : f32
// CHECK:                 %[[VAL_114:.*]] = fir.load %[[VAL_14]] {tbaa = [#[[GLBL_DYINV_TAG]]]} : !fir.ref<f32>
// CHECK:                 %[[VAL_115:.*]] = arith.mulf %[[VAL_113]], %[[VAL_114]] fastmath<contract> : f32
// CHECK:                 %[[VAL_116:.*]] = arith.subi %[[VAL_90]], %[[VAL_6]] : i32
// CHECK:                 %[[VAL_117:.*]] = fir.convert %[[VAL_116]] : (i32) -> i64
// CHECK:                 %[[VAL_118:.*]] = fir.array_coor %[[VAL_37]] %[[VAL_89]], %[[VAL_81]], %[[VAL_117]] : (!fir.box<!fir.array<?x?x?xf32>>, i64, i64, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_119:.*]] = fir.load %[[VAL_118]] {tbaa = [#[[ARG_Y_TAG]]]} : !fir.ref<f32>
// CHECK:                 %[[VAL_120:.*]] = fir.array_coor %[[VAL_37]] %[[VAL_89]], %[[VAL_81]], %[[VAL_91]] : (!fir.box<!fir.array<?x?x?xf32>>, i64, i64, i64) -> !fir.ref<f32>
// CHECK:                 %[[VAL_121:.*]] = fir.load %[[VAL_120]] {tbaa = [#[[ARG_Y_TAG]]]} : !fir.ref<f32>
// CHECK:                 %[[VAL_122:.*]] = arith.subf %[[VAL_119]], %[[VAL_121]] fastmath<contract> : f32
// CHECK:                 %[[VAL_123:.*]] = fir.no_reassoc %[[VAL_122]] : f32
// load from local allocation
// CHECK:                 %[[VAL_124:.*]] = fir.load %[[VAL_28]] : !fir.ref<f32>
// CHECK:                 %[[VAL_125:.*]] = arith.mulf %[[VAL_123]], %[[VAL_124]] fastmath<contract> : f32
// CHECK:                 %[[VAL_126:.*]] = arith.addf %[[VAL_115]], %[[VAL_125]] fastmath<contract> : f32
// CHECK:                 %[[VAL_127:.*]] = fir.no_reassoc %[[VAL_126]] : f32
// CHECK:                 %[[VAL_128:.*]] = arith.mulf %[[VAL_105]], %[[VAL_127]] fastmath<contract> : f32
// CHECK:                 %[[VAL_129:.*]] = arith.addf %[[VAL_99]], %[[VAL_128]] fastmath<contract> : f32
// CHECK:                 fir.store %[[VAL_129]] to %[[VAL_97]] {tbaa = [#[[ARG_LOW_TAG]]]} : !fir.ref<f32>
// CHECK:                 %[[VAL_130:.*]] = arith.addi %[[VAL_63]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_131:.*]] = fir.convert %[[VAL_5]] : (index) -> i32
// load from local allocation
// CHECK:                 %[[VAL_132:.*]] = fir.load %[[VAL_30]] : !fir.ref<i32>
// CHECK:                 %[[VAL_133:.*]] = arith.addi %[[VAL_132]], %[[VAL_131]] : i32
// CHECK:                 fir.result %[[VAL_130]], %[[VAL_133]] : index, i32
// CHECK:               }
// store to local allocation
// CHECK:               fir.store %[[VAL_134:.*]]#1 to %[[VAL_30]] : !fir.ref<i32>
// CHECK:               %[[VAL_135:.*]] = arith.addi %[[VAL_56]], %[[VAL_5]] : index
// CHECK:               %[[VAL_136:.*]] = fir.convert %[[VAL_5]] : (index) -> i32
// local allocation:
// CHECK:               %[[VAL_137:.*]] = fir.load %[[VAL_32]] : !fir.ref<i32>
// CHECK:               %[[VAL_138:.*]] = arith.addi %[[VAL_137]], %[[VAL_136]] : i32
// CHECK:               fir.result %[[VAL_135]], %[[VAL_138]] : index, i32
// CHECK:             }
// local allocation:
// CHECK:             fir.store %[[VAL_139:.*]]#1 to %[[VAL_32]] : !fir.ref<i32>
// CHECK:             %[[VAL_140:.*]] = arith.addi %[[VAL_47]], %[[VAL_5]] : index
// CHECK:             %[[VAL_141:.*]] = fir.convert %[[VAL_5]] : (index) -> i32
// local allocation:
// CHECK:             %[[VAL_142:.*]] = fir.load %[[VAL_34]] : !fir.ref<i32>
// CHECK:             %[[VAL_143:.*]] = arith.addi %[[VAL_142]], %[[VAL_141]] : i32
// CHECK:             fir.result %[[VAL_140]], %[[VAL_143]] : index, i32
// CHECK:           }
// local allocation:
// CHECK:           fir.store %[[VAL_144:.*]]#1 to %[[VAL_34]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

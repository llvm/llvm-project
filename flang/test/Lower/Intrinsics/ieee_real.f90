! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program p
  use ieee_arithmetic, only: ieee_real

  ! CHECK:     %[[V_0:[0-9]+]] = fir.alloca i16 {bindc_name = "j2", uniq_name = "_QFEj2"}
  ! CHECK:     %[[V_1:[0-9]+]]:2 = hlfir.declare %[[V_0]] {uniq_name = "_QFEj2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
  ! CHECK:     %[[V_2:[0-9]+]] = fir.alloca i64 {bindc_name = "j8", uniq_name = "_QFEj8"}
  ! CHECK:     %[[V_3:[0-9]+]]:2 = hlfir.declare %[[V_2]] {uniq_name = "_QFEj8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK:     %[[V_4:[0-9]+]] = fir.alloca f16 {bindc_name = "x2", uniq_name = "_QFEx2"}
  ! CHECK:     %[[V_5:[0-9]+]]:2 = hlfir.declare %[[V_4]] {uniq_name = "_QFEx2"} : (!fir.ref<f16>) -> (!fir.ref<f16>, !fir.ref<f16>)
  ! CHECK:     %[[V_6:[0-9]+]] = fir.alloca f32 {bindc_name = "x4", uniq_name = "_QFEx4"}
  ! CHECK:     %[[V_7:[0-9]+]]:2 = hlfir.declare %[[V_6]] {uniq_name = "_QFEx4"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK:     %[[V_8:[0-9]+]] = fir.alloca f64 {bindc_name = "x8", uniq_name = "_QFEx8"}
  ! CHECK:     %[[V_9:[0-9]+]]:2 = hlfir.declare %[[V_8]] {uniq_name = "_QFEx8"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  integer(2) :: j2
  integer(8) :: j8
  real(2) ::  x2
  real(4) ::  x4
  real(8) ::  x8

  ! CHECK:     hlfir.assign %c-32768{{.*}} to %[[V_1]]#0 : i16, !fir.ref<i16>
  j2 = -huge(j2) - 1

  ! CHECK:     %[[V_10:[0-9]+]] = fir.load %[[V_1]]#0 : !fir.ref<i16>
  ! CHECK:     %[[V_11:[0-9]+]] = fir.convert %[[V_10]] : (i16) -> f32
  ! CHECK:     hlfir.assign %[[V_11]] to %[[V_7]]#0 : f32, !fir.ref<f32>
  x4 = ieee_real(j2,4) ! exact
! print*, j2, ' -> ', x4

  ! CHECK:     hlfir.assign %c33{{.*}} to %[[V_3]]#0 : i64, !fir.ref<i64>
  j8 = 33

  ! CHECK:     %[[V_12:[0-9]+]] = fir.load %[[V_3]]#0 : !fir.ref<i64>
  ! CHECK:     %[[V_13:[0-9]+]] = fir.convert %[[V_12]] : (i64) -> f32
  ! CHECK:     %[[V_14:[0-9]+]] = fir.convert %[[V_13]] : (f32) -> i64
  ! CHECK:     %[[V_15:[0-9]+]] = arith.cmpi eq, %[[V_12]], %[[V_14]] : i64
  ! CHECK:     %[[V_16:[0-9]+]] = fir.if %[[V_15]] -> (f32) {
  ! CHECK:       fir.result %[[V_13]] : f32
  ! CHECK:     } else {
  ! CHECK:       %[[V_27:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK-DAG:   %[[V_28:[0-9]+]] = arith.cmpi slt, %[[V_12]], %c0{{.*}} : i64
  ! CHECK-DAG:   %[[V_29:[0-9]+]] = arith.cmpi sgt, %[[V_12]], %c0{{.*}} : i64
  ! CHECK-DAG:   %[[V_30:[0-9]+]] = arith.bitcast %[[V_13]] : f32 to i32
  ! CHECK-DAG:   %[[V_31:[0-9]+]] = arith.andi %[[V_30]], %c1{{.*}} : i32
  ! CHECK-DAG:   %[[V_32:[0-9]+]] = fir.convert %[[V_31]] : (i32) -> i1
  ! CHECK-DAG:   %[[V_33:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c5{{.*}} : i32
  ! CHECK-DAG:   %[[V_34:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c1{{.*}} : i32
  ! CHECK-DAG:   %[[V_35:[0-9]+]] = arith.ori %[[V_34]], %[[V_33]] : i1
  ! CHECK-DAG:   %[[V_36:[0-9]+]] = arith.andi %[[V_35]], %[[V_32]] : i1
  ! CHECK-DAG:   %[[V_37:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c0{{.*}} : i32
  ! CHECK-DAG:   %[[V_38:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c4{{.*}} : i32
  ! CHECK-DAG:   %[[V_39:[0-9]+]] = arith.cmpi slt, %[[V_12]], %[[V_14]] : i64
  ! CHECK-DAG:   %[[V_40:[0-9]+]] = arith.addi %[[V_30]], %c1{{.*}} : i32
  ! CHECK-DAG:   %[[V_41:[0-9]+]] = arith.subi %[[V_30]], %c1{{.*}} : i32
  ! CHECK:       %[[V_42:[0-9]+]] = fir.if %[[V_39]] -> (f32) {
  ! CHECK-DAG:     %[[V_44:[0-9]+]] = arith.andi %[[V_37]], %[[V_29]] : i1
  ! CHECK-DAG:     %[[V_45:[0-9]+]] = arith.andi %[[V_38]], %[[V_28]] : i1
  ! CHECK-DAG:     %[[V_46:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c3{{.*}} : i32
  ! CHECK-DAG:     %[[V_47:[0-9]+]] = arith.ori %[[V_36]], %[[V_44]] : i1
  ! CHECK-DAG:     %[[V_48:[0-9]+]] = arith.ori %[[V_47]], %[[V_45]] : i1
  ! CHECK-DAG:     %[[V_49:[0-9]+]] = arith.ori %[[V_48]], %[[V_46]] : i1
  ! CHECK:         %[[V_50:[0-9]+]] = fir.if %[[V_49]] -> (f32) {
  ! CHECK:           %[[V_51:[0-9]+]] = arith.select %[[V_28]], %[[V_40]], %[[V_41]] : i32
  ! CHECK:           %[[V_52:[0-9]+]] = arith.bitcast %[[V_51]] : i32 to f32
  ! CHECK:           fir.result %[[V_52]] : f32
  ! CHECK:         } else {
  ! CHECK:           fir.result %[[V_13]] : f32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_50]] : f32
  ! CHECK:       } else {
  ! CHECK-DAG:     %[[V_44:[0-9]+]] = arith.andi %[[V_37]], %[[V_28]] : i1
  ! CHECK-DAG:     %[[V_45:[0-9]+]] = arith.andi %[[V_38]], %[[V_29]] : i1
  ! CHECK-DAG:     %[[V_46:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c2{{.*}} : i32
  ! CHECK-DAG:     %[[V_47:[0-9]+]] = arith.ori %[[V_36]], %[[V_44]] : i1
  ! CHECK-DAG:     %[[V_48:[0-9]+]] = arith.ori %[[V_47]], %[[V_45]] : i1
  ! CHECK-DAG:     %[[V_49:[0-9]+]] = arith.ori %[[V_48]], %[[V_46]] : i1
  ! CHECK:         %[[V_50:[0-9]+]] = fir.if %[[V_49]] -> (f32) {
  ! CHECK:           %[[V_51:[0-9]+]] = arith.select %[[V_29]], %[[V_40]], %[[V_41]] : i32
  ! CHECK:           %[[V_52:[0-9]+]] = arith.bitcast %[[V_51]] : i32 to f32
  ! CHECK:           fir.result %[[V_52]] : f32
  ! CHECK:         } else {
  ! CHECK:           fir.result %[[V_13]] : f32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_50]] : f32
  ! CHECK:       }
  ! CHECK:       %[[V_43:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_42]]) <{bit = 516 : i32}> : (f32) -> i1
  ! CHECK:       fir.if %[[V_43]] {
  ! CHECK:         %[[V_44:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_45:[0-9]+]] = fir.call @feraiseexcept(%[[V_44]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_44:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_42]]) <{bit = 240 : i32}> : (f32) -> i1
  ! CHECK:         fir.if %[[V_44]] {
  ! CHECK:           %[[V_45:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:           %[[V_46:[0-9]+]] = fir.call @feraiseexcept(%[[V_45]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         } else {
  ! CHECK:           %[[V_45:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:           %[[V_46:[0-9]+]] = fir.call @feraiseexcept(%[[V_45]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_42]] : f32
  ! CHECK:     }
  ! CHECK:     hlfir.assign %[[V_16]] to %[[V_7]]#0 : f32, !fir.ref<f32>
  x4 = ieee_real(j8,4)
! print*, j8, ' -> ', x4

  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_5]]#0 : f16, !fir.ref<f16>
  x2 = 3.33

  ! CHECK:     %[[V_17:[0-9]+]] = fir.load %[[V_5]]#0 : !fir.ref<f16>
  ! CHECK:     %[[V_18:[0-9]+]] = fir.convert %[[V_17]] : (f16) -> f32
  ! CHECK:     %[[V_19:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_18]]) <{bit = 1 : i32}> : (f32) -> i1
  ! CHECK:     %[[V_20:[0-9]+]] = fir.if %[[V_19]] -> (f32) {
  ! CHECK:       %[[V_27:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_28:[0-9]+]] = fir.call @feraiseexcept(%[[V_27]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_29:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_4) : !fir.ref<!fir.array<12xi32>>
  ! CHECK:       %[[V_30:[0-9]+]] = fir.coordinate_of %[[V_29]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
  ! CHECK:       %[[V_31:[0-9]+]] = fir.load %[[V_30]] : !fir.ref<i32>
  ! CHECK:       %[[V_32:[0-9]+]] = arith.bitcast %[[V_31]] : i32 to f32
  ! CHECK:       fir.result %[[V_32]] : f32
  ! CHECK:     } else {
  ! CHECK:       fir.result %[[V_18]] : f32
  ! CHECK:     }
  ! CHECK:     %[[V_21:[0-9]+]] = fir.convert %[[V_20]] : (f32) -> f16
  ! CHECK:     hlfir.assign %[[V_21]] to %[[V_5]]#0 : f16, !fir.ref<f16>
  x2 = ieee_real(x2,4) ! exact
! print*, x2, ' -> ', x2

  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_9]]#0 : f64, !fir.ref<f64>
  x8 = -0.

  ! CHECK:     %[[V_22:[0-9]+]] = fir.load %[[V_9]]#0 : !fir.ref<f64>
  ! CHECK:     %[[V_23:[0-9]+]] = fir.convert %[[V_22]] : (f64) -> f32
  ! CHECK:     %[[V_24:[0-9]+]] = fir.convert %[[V_23]] : (f32) -> f64
  ! CHECK:     %[[V_25:[0-9]+]] = arith.cmpf ueq, %[[V_22]], %[[V_24]] fastmath<contract> : f64
  ! CHECK:     %[[V_26:[0-9]+]] = fir.if %[[V_25]] -> (f32) {
  ! CHECK:       %[[V_27:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_23]]) <{bit = 1 : i32}> : (f32) -> i1
  ! CHECK:       %[[V_28:[0-9]+]] = fir.if %[[V_27]] -> (f32) {
  ! CHECK:         %[[V_29:[0-9]+]] = fir.call @_FortranAMapException(%c1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_30:[0-9]+]] = fir.call @feraiseexcept(%[[V_29]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_31:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_4) : !fir.ref<!fir.array<12xi32>>
  ! CHECK:         %[[V_32:[0-9]+]] = fir.coordinate_of %[[V_31]], %c2{{.*}} : (!fir.ref<!fir.array<12xi32>>, i8) -> !fir.ref<i32>
  ! CHECK:         %[[V_33:[0-9]+]] = fir.load %[[V_32]] : !fir.ref<i32>
  ! CHECK:         %[[V_34:[0-9]+]] = arith.bitcast %[[V_33]] : i32 to f32
  ! CHECK:         fir.result %[[V_34]] : f32
  ! CHECK:       } else {
  ! CHECK:         fir.result %[[V_23]] : f32
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_28]] : f32
  ! CHECK:     } else {
  ! CHECK-DAG:   %[[V_27:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK-DAG:   %[[V_28:[0-9]+]] = arith.cmpf olt, %[[V_22]], %cst{{[_0-9]*}} fastmath<contract> : f64
  ! CHECK-DAG:   %[[V_29:[0-9]+]] = arith.cmpf ogt, %[[V_22]], %cst{{[_0-9]*}} fastmath<contract> : f64
  ! CHECK-DAG:   %[[V_30:[0-9]+]] = arith.bitcast %[[V_23]] : f32 to i32
  ! CHECK-DAG:   %[[V_31:[0-9]+]] = arith.andi %[[V_30]], %c1{{.*}} : i32
  ! CHECK-DAG:   %[[V_32:[0-9]+]] = fir.convert %[[V_31]] : (i32) -> i1
  ! CHECK-DAG:   %[[V_33:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c5{{.*}} : i32
  ! CHECK-DAG:   %[[V_34:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c1{{.*}} : i32
  ! CHECK-DAG:   %[[V_35:[0-9]+]] = arith.ori %[[V_34]], %[[V_33]] : i1
  ! CHECK-DAG:   %[[V_36:[0-9]+]] = arith.andi %[[V_35]], %[[V_32]] : i1
  ! CHECK-DAG:   %[[V_37:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c0{{.*}} : i32
  ! CHECK-DAG:   %[[V_38:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c4{{.*}} : i32
  ! CHECK-DAG:   %[[V_39:[0-9]+]] = arith.cmpf olt, %[[V_22]], %[[V_24]] fastmath<contract> : f64
  ! CHECK-DAG:   %[[V_40:[0-9]+]] = arith.addi %[[V_30]], %c1{{.*}} : i32
  ! CHECK-DAG:   %[[V_41:[0-9]+]] = arith.subi %[[V_30]], %c1{{.*}} : i32
  ! CHECK:       %[[V_42:[0-9]+]] = fir.if %[[V_39]] -> (f32) {
  ! CHECK-DAG:     %[[V_44:[0-9]+]] = arith.andi %[[V_37]], %[[V_29]] : i1
  ! CHECK-DAG:     %[[V_45:[0-9]+]] = arith.andi %[[V_38]], %[[V_28]] : i1
  ! CHECK-DAG:     %[[V_46:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c3{{.*}} : i32
  ! CHECK-DAG:     %[[V_47:[0-9]+]] = arith.ori %[[V_36]], %[[V_44]] : i1
  ! CHECK-DAG:     %[[V_48:[0-9]+]] = arith.ori %[[V_47]], %[[V_45]] : i1
  ! CHECK-DAG:     %[[V_49:[0-9]+]] = arith.ori %[[V_48]], %[[V_46]] : i1
  ! CHECK:         %[[V_50:[0-9]+]] = fir.if %[[V_49]] -> (f32) {
  ! CHECK:           %[[V_51:[0-9]+]] = arith.select %[[V_28]], %[[V_40]], %[[V_41]] : i32
  ! CHECK:           %[[V_52:[0-9]+]] = arith.bitcast %[[V_51]] : i32 to f32
  ! CHECK:           fir.result %[[V_52]] : f32
  ! CHECK:         } else {
  ! CHECK:           fir.result %[[V_23]] : f32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_50]] : f32
  ! CHECK:       } else {
  ! CHECK-DAG:     %[[V_44:[0-9]+]] = arith.andi %[[V_37]], %[[V_28]] : i1
  ! CHECK-DAG:     %[[V_45:[0-9]+]] = arith.andi %[[V_38]], %[[V_29]] : i1
  ! CHECK-DAG:     %[[V_46:[0-9]+]] = arith.cmpi eq, %[[V_27]], %c2{{.*}} : i32
  ! CHECK-DAG:     %[[V_47:[0-9]+]] = arith.ori %[[V_36]], %[[V_44]] : i1
  ! CHECK-DAG:     %[[V_48:[0-9]+]] = arith.ori %[[V_47]], %[[V_45]] : i1
  ! CHECK-DAG:     %[[V_49:[0-9]+]] = arith.ori %[[V_48]], %[[V_46]] : i1
  ! CHECK:         %[[V_50:[0-9]+]] = fir.if %[[V_49]] -> (f32) {
  ! CHECK:           %[[V_51:[0-9]+]] = arith.select %[[V_29]], %[[V_40]], %[[V_41]] : i32
  ! CHECK:           %[[V_52:[0-9]+]] = arith.bitcast %[[V_51]] : i32 to f32
  ! CHECK:           fir.result %[[V_52]] : f32
  ! CHECK:         } else {
  ! CHECK:           fir.result %[[V_23]] : f32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_50]] : f32
  ! CHECK:       }
  ! CHECK:       %[[V_43:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_42]]) <{bit = 516 : i32}> : (f32) -> i1
  ! CHECK:       fir.if %[[V_43]] {
  ! CHECK:         %[[V_44:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:         %[[V_45:[0-9]+]] = fir.call @feraiseexcept(%[[V_44]]) fastmath<contract> : (i32) -> i32
  ! CHECK:       } else {
  ! CHECK:         %[[V_44:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_42]]) <{bit = 240 : i32}> : (f32) -> i1
  ! CHECK:         fir.if %[[V_44]] {
  ! CHECK:           %[[V_45:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:           %[[V_46:[0-9]+]] = fir.call @feraiseexcept(%[[V_45]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         } else {
  ! CHECK:           %[[V_45:[0-9]+]] = fir.call @_FortranAMapException(%c32{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:           %[[V_46:[0-9]+]] = fir.call @feraiseexcept(%[[V_45]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_42]] : f32
  ! CHECK:     }
  ! CHECK:     hlfir.assign %[[V_26]] to %[[V_7]]#0 : f32, !fir.ref<f32>
  x4 = ieee_real(x8,4)
! print*, x8, ' -> ', x4
end

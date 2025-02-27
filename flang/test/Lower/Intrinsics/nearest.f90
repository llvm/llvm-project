! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! CHECK-LABEL: c.func @_QPnearest_test1
  ! CHECK:     %[[V_0:[0-9]+]] = fir.dummy_scope : !fir.dscope
  ! CHECK:     %[[V_1:[0-9]+]] = fir.alloca f16 {bindc_name = "res", uniq_name = "_QFnearest_test1Eres"}
  ! CHECK:     %[[V_2:[0-9]+]] = fir.declare %[[V_1]] {uniq_name = "_QFnearest_test1Eres"} : (!fir.ref<f16>) -> !fir.ref<f16>
  ! CHECK:     %[[V_3:[0-9]+]] = fir.declare %arg1 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test1Es"} : (!fir.ref<f16>, !fir.dscope) -> !fir.ref<f16>
  ! CHECK:     %[[V_4:[0-9]+]] = fir.declare %arg0 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test1Ex"} : (!fir.ref<f16>, !fir.dscope) -> !fir.ref<f16>
  ! CHECK:     %[[V_5:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f16>
  ! CHECK:     %[[V_6:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f16>
  ! CHECK:     %[[V_7:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 3 : i32}> : (f16) -> i1
  ! CHECK:     %[[V_8:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_6]]) <{bit = 96 : i32}> : (f16) -> i1
  ! CHECK:     fir.if %[[V_8]] {
  ! CHECK:       fir.call @_FortranAReportFatalUserError
  ! CHECK:     }
  ! CHECK:     %[[V_9:[0-9]+]] = arith.bitcast %[[V_6]] : f16 to i16
  ! CHECK:     %[[V_10:[0-9]+]] = arith.shrui %[[V_9]], %c15{{.*}} : i16
  ! CHECK:     %[[V_11:[0-9]+]] = arith.cmpi ne, %[[V_10]], %c1{{.*}} : i16
  ! CHECK:     %[[V_12:[0-9]+]] = arith.bitcast %[[V_5]] : f16 to i16
  ! CHECK:     %[[V_13:[0-9]+]] = arith.shrui %[[V_12]], %c15{{.*}} : i16
  ! CHECK:     %[[V_14:[0-9]+]] = fir.convert %[[V_13]] : (i16) -> i1
  ! CHECK:     %[[V_15:[0-9]+]] = arith.cmpi ne, %[[V_11]], %[[V_14]] : i1
  ! CHECK:     %[[V_16:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 516 : i32}> : (f16) -> i1
  ! CHECK:     %[[V_17:[0-9]+]] = arith.andi %[[V_16]], %[[V_15]] : i1
  ! CHECK:     %[[V_18:[0-9]+]] = arith.ori %[[V_7]], %[[V_17]] : i1
  ! CHECK:     %[[V_19:[0-9]+]] = fir.if %[[V_18]] -> (f16) {
  ! CHECK:       fir.result %[[V_5]] : f16
  ! CHECK:     } else {
  ! CHECK:       %[[V_20:[0-9]+]] = arith.cmpf oeq, %[[V_5]], %cst{{[_0-9]*}} fastmath<contract> : f16
  ! CHECK:       %[[V_21:[0-9]+]] = fir.if %[[V_20]] -> (f16) {
  ! CHECK:         %[[V_22:[0-9]+]] = arith.select %[[V_11]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : f16
  ! CHECK:         %[[V_23:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                            fir.call @feraiseexcept(%[[V_23]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         fir.result %[[V_22]] : f16
  ! CHECK:       } else {
  ! CHECK-DAG:     %[[V_22:[0-9]+]] = arith.subi %[[V_12]], %c1{{.*}} : i16
  ! CHECK-DAG:     %[[V_23:[0-9]+]] = arith.addi %[[V_12]], %c1{{.*}} : i16
  ! CHECK:         %[[V_24:[0-9]+]] = arith.select %[[V_15]], %[[V_23]], %[[V_22]] : i16
  ! CHECK:         %[[V_25:[0-9]+]] = arith.bitcast %[[V_24]] : i16 to f16
  ! CHECK:         %[[V_26:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_25]]) <{bit = 516 : i32}> : (f16) -> i1
  ! CHECK:         fir.if %[[V_26]] {
  ! CHECK:           %[[V_28:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                              fir.call @feraiseexcept(%[[V_28]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         %[[V_27:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_25]]) <{bit = 144 : i32}> : (f16) -> i1
  ! CHECK:         fir.if %[[V_27]] {
  ! CHECK:           %[[V_28:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                              fir.call @feraiseexcept(%[[V_28]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_25]] : f16
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_21]] : f16
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_19]] to %[[V_2]] : !fir.ref<f16>
  ! CHECK:     return
  ! CHECK:   }
subroutine nearest_test1(x, s)
  real(kind=2) :: x, s, res
  res = nearest(x, s)
end

! CHECK-LABEL: c.func @_QPnearest_test2
  ! CHECK:     %[[V_0:[0-9]+]] = fir.dummy_scope : !fir.dscope
  ! CHECK:     %[[V_1:[0-9]+]] = fir.alloca bf16 {bindc_name = "res", uniq_name = "_QFnearest_test2Eres"}
  ! CHECK:     %[[V_2:[0-9]+]] = fir.declare %[[V_1]] {uniq_name = "_QFnearest_test2Eres"} : (!fir.ref<bf16>) -> !fir.ref<bf16>
  ! CHECK:     %[[V_3:[0-9]+]] = fir.declare %arg1 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test2Es"} : (!fir.ref<bf16>, !fir.dscope) -> !fir.ref<bf16>
  ! CHECK:     %[[V_4:[0-9]+]] = fir.declare %arg0 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test2Ex"} : (!fir.ref<bf16>, !fir.dscope) -> !fir.ref<bf16>
  ! CHECK:     %[[V_5:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<bf16>
  ! CHECK:     %[[V_6:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<bf16>
  ! CHECK:     %[[V_7:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 3 : i32}> : (bf16) -> i1
  ! CHECK:     %[[V_8:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_6]]) <{bit = 96 : i32}> : (bf16) -> i1
  ! CHECK:     fir.if %[[V_8]] {
  ! CHECK:       fir.call @_FortranAReportFatalUserError
  ! CHECK:     }
  ! CHECK:     %[[V_9:[0-9]+]] = fir.convert %[[V_6]] : (bf16) -> f32
  ! CHECK:     %[[V_10:[0-9]+]] = arith.bitcast %[[V_9]] : f32 to i32
  ! CHECK:     %[[V_11:[0-9]+]] = arith.shrui %[[V_10]], %c31{{.*}} : i32
  ! CHECK:     %[[V_12:[0-9]+]] = fir.convert %[[V_11]] : (i32) -> i16
  ! CHECK:     %[[V_13:[0-9]+]] = arith.cmpi ne, %[[V_12]], %c1{{.*}} : i16
  ! CHECK:     %[[V_14:[0-9]+]] = fir.convert %[[V_5]] : (bf16) -> f32
  ! CHECK:     %[[V_15:[0-9]+]] = arith.bitcast %[[V_14]] : f32 to i32
  ! CHECK:     %[[V_16:[0-9]+]] = arith.shrui %[[V_15]], %c31{{.*}} : i32
  ! CHECK:     %[[V_17:[0-9]+]] = fir.convert %[[V_16]] : (i32) -> i1
  ! CHECK:     %[[V_18:[0-9]+]] = arith.cmpi ne, %[[V_13]], %[[V_17]] : i1
  ! CHECK:     %[[V_19:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 516 : i32}> : (bf16) -> i1
  ! CHECK:     %[[V_20:[0-9]+]] = arith.andi %[[V_19]], %[[V_18]] : i1
  ! CHECK:     %[[V_21:[0-9]+]] = arith.ori %[[V_7]], %[[V_20]] : i1
  ! CHECK:     %[[V_22:[0-9]+]] = fir.if %[[V_21]] -> (bf16) {
  ! CHECK:       fir.result %[[V_5]] : bf16
  ! CHECK:     } else {
  ! CHECK:       %[[V_23:[0-9]+]] = arith.cmpf oeq, %[[V_5]], %cst{{[_0-9]*}} fastmath<contract> : bf16
  ! CHECK:       %[[V_24:[0-9]+]] = fir.if %[[V_23]] -> (bf16) {
  ! CHECK:         %[[V_25:[0-9]+]] = arith.select %[[V_13]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : bf16
  ! CHECK:         %[[V_26:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                            fir.call @feraiseexcept(%[[V_26]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         fir.result %[[V_25]] : bf16
  ! CHECK:       } else {
  ! CHECK:         %[[V_25:[0-9]+]] = arith.bitcast %[[V_5]] : bf16 to i16
  ! CHECK-DAG:     %[[V_26:[0-9]+]] = arith.subi %[[V_25]], %c1{{.*}} : i16
  ! CHECK-DAG:     %[[V_27:[0-9]+]] = arith.addi %[[V_25]], %c1{{.*}} : i16
  ! CHECK:         %[[V_28:[0-9]+]] = arith.select %[[V_18]], %[[V_27]], %[[V_26]] : i16
  ! CHECK:         %[[V_29:[0-9]+]] = arith.bitcast %[[V_28]] : i16 to bf16
  ! CHECK:         %[[V_30:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_29]]) <{bit = 516 : i32}> : (bf16) -> i1
  ! CHECK:         fir.if %[[V_30]] {
  ! CHECK:           %[[V_32:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                              fir.call @feraiseexcept(%[[V_32]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         %[[V_31:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_29]]) <{bit = 144 : i32}> : (bf16) -> i1
  ! CHECK:         fir.if %[[V_31]] {
  ! CHECK:           %[[V_32:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                              fir.call @feraiseexcept(%[[V_32]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_29]] : bf16
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_24]] : bf16
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_22]] to %[[V_2]] : !fir.ref<bf16>
  ! CHECK:     return
  ! CHECK:   }
subroutine nearest_test2(x, s)
  real(kind=3) :: x, s, res
  res = nearest(x, s)
end

! CHECK-LABEL: c.func @_QPnearest_test3
  ! CHECK:     %[[V_0:[0-9]+]] = fir.dummy_scope : !fir.dscope
  ! CHECK:     %[[V_1:[0-9]+]] = fir.alloca f32 {bindc_name = "res", uniq_name = "_QFnearest_test3Eres"}
  ! CHECK:     %[[V_2:[0-9]+]] = fir.declare %[[V_1]] {uniq_name = "_QFnearest_test3Eres"} : (!fir.ref<f32>) -> !fir.ref<f32>
  ! CHECK:     %[[V_3:[0-9]+]] = fir.declare %arg1 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test3Es"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  ! CHECK:     %[[V_4:[0-9]+]] = fir.declare %arg0 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test3Ex"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  ! CHECK:     %[[V_5:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f32>
  ! CHECK:     %[[V_6:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f32>
  ! CHECK:     %[[V_7:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 3 : i32}> : (f32) -> i1
  ! CHECK:     %[[V_8:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_6]]) <{bit = 96 : i32}> : (f32) -> i1
  ! CHECK:     fir.if %[[V_8]] {
  ! CHECK:       fir.call @_FortranAReportFatalUserError
  ! CHECK:     }
  ! CHECK:     %[[V_9:[0-9]+]] = arith.bitcast %[[V_6]] : f32 to i32
  ! CHECK:     %[[V_10:[0-9]+]] = arith.shrui %[[V_9]], %c31{{.*}} : i32
  ! CHECK:     %[[V_11:[0-9]+]] = arith.cmpi ne, %[[V_10]], %c1{{.*}} : i32
  ! CHECK:     %[[V_12:[0-9]+]] = arith.bitcast %[[V_5]] : f32 to i32
  ! CHECK:     %[[V_13:[0-9]+]] = arith.shrui %[[V_12]], %c31{{.*}} : i32
  ! CHECK:     %[[V_14:[0-9]+]] = fir.convert %[[V_13]] : (i32) -> i1
  ! CHECK:     %[[V_15:[0-9]+]] = arith.cmpi ne, %[[V_11]], %[[V_14]] : i1
  ! CHECK:     %[[V_16:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 516 : i32}> : (f32) -> i1
  ! CHECK:     %[[V_17:[0-9]+]] = arith.andi %[[V_16]], %[[V_15]] : i1
  ! CHECK:     %[[V_18:[0-9]+]] = arith.ori %[[V_7]], %[[V_17]] : i1
  ! CHECK:     %[[V_19:[0-9]+]] = fir.if %[[V_18]] -> (f32) {
  ! CHECK:       fir.result %[[V_5]] : f32
  ! CHECK:     } else {
  ! CHECK:       %[[V_20:[0-9]+]] = arith.cmpf oeq, %[[V_5]], %cst{{[_0-9]*}} fastmath<contract> : f32
  ! CHECK:       %[[V_21:[0-9]+]] = fir.if %[[V_20]] -> (f32) {
  ! CHECK:         %[[V_22:[0-9]+]] = arith.select %[[V_11]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : f32
  ! CHECK:         %[[V_23:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                            fir.call @feraiseexcept(%[[V_23]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         fir.result %[[V_22]] : f32
  ! CHECK:       } else {
  ! CHECK-DAG:     %[[V_22:[0-9]+]] = arith.subi %[[V_12]], %c1{{.*}} : i32
  ! CHECK-DAG:     %[[V_23:[0-9]+]] = arith.addi %[[V_12]], %c1{{.*}} : i32
  ! CHECK:         %[[V_24:[0-9]+]] = arith.select %[[V_15]], %[[V_23]], %[[V_22]] : i32
  ! CHECK:         %[[V_25:[0-9]+]] = arith.bitcast %[[V_24]] : i32 to f32
  ! CHECK:         %[[V_26:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_25]]) <{bit = 516 : i32}> : (f32) -> i1
  ! CHECK:         fir.if %[[V_26]] {
  ! CHECK:           %[[V_28:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                              fir.call @feraiseexcept(%[[V_28]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         %[[V_27:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_25]]) <{bit = 144 : i32}> : (f32) -> i1
  ! CHECK:         fir.if %[[V_27]] {
  ! CHECK:           %[[V_28:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                              fir.call @feraiseexcept(%[[V_28]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_25]] : f32
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_21]] : f32
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_19]] to %[[V_2]] : !fir.ref<f32>
  ! CHECK:     return
  ! CHECK:   }
subroutine nearest_test3(x, s)
  real :: x, s, res
  res = nearest(x, s)
end

! CHECK-LABEL: c.func @_QPnearest_test4
  ! CHECK:     %[[V_0:[0-9]+]] = fir.dummy_scope : !fir.dscope
  ! CHECK:     %[[V_1:[0-9]+]] = fir.alloca f64 {bindc_name = "res", uniq_name = "_QFnearest_test4Eres"}
  ! CHECK:     %[[V_2:[0-9]+]] = fir.declare %[[V_1]] {uniq_name = "_QFnearest_test4Eres"} : (!fir.ref<f64>) -> !fir.ref<f64>
  ! CHECK:     %[[V_3:[0-9]+]] = fir.declare %arg1 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test4Es"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
  ! CHECK:     %[[V_4:[0-9]+]] = fir.declare %arg0 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test4Ex"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
  ! CHECK:     %[[V_5:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f64>
  ! CHECK:     %[[V_6:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f64>
  ! CHECK:     %[[V_7:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 3 : i32}> : (f64) -> i1
  ! CHECK:     %[[V_8:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_6]]) <{bit = 96 : i32}> : (f64) -> i1
  ! CHECK:     fir.if %[[V_8]] {
  ! CHECK:       fir.call @_FortranAReportFatalUserError
  ! CHECK:     }
  ! CHECK:     %[[V_9:[0-9]+]] = arith.bitcast %[[V_6]] : f64 to i64
  ! CHECK:     %[[V_10:[0-9]+]] = arith.shrui %[[V_9]], %c63{{.*}} : i64
  ! CHECK:     %[[V_11:[0-9]+]] = arith.cmpi ne, %[[V_10]], %c1{{.*}} : i64
  ! CHECK:     %[[V_12:[0-9]+]] = arith.bitcast %[[V_5]] : f64 to i64
  ! CHECK:     %[[V_13:[0-9]+]] = arith.shrui %[[V_12]], %c63{{.*}} : i64
  ! CHECK:     %[[V_14:[0-9]+]] = fir.convert %[[V_13]] : (i64) -> i1
  ! CHECK:     %[[V_15:[0-9]+]] = arith.cmpi ne, %[[V_11]], %[[V_14]] : i1
  ! CHECK:     %[[V_16:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 516 : i32}> : (f64) -> i1
  ! CHECK:     %[[V_17:[0-9]+]] = arith.andi %[[V_16]], %[[V_15]] : i1
  ! CHECK:     %[[V_18:[0-9]+]] = arith.ori %[[V_7]], %[[V_17]] : i1
  ! CHECK:     %[[V_19:[0-9]+]] = fir.if %[[V_18]] -> (f64) {
  ! CHECK:       fir.result %[[V_5]] : f64
  ! CHECK:     } else {
  ! CHECK:       %[[V_20:[0-9]+]] = arith.cmpf oeq, %[[V_5]], %cst{{[_0-9]*}} fastmath<contract> : f64
  ! CHECK:       %[[V_21:[0-9]+]] = fir.if %[[V_20]] -> (f64) {
  ! CHECK:         %[[V_22:[0-9]+]] = arith.select %[[V_11]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : f64
  ! CHECK:         %[[V_23:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                            fir.call @feraiseexcept(%[[V_23]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         fir.result %[[V_22]] : f64
  ! CHECK:       } else {
  ! CHECK-DAG:     %[[V_22:[0-9]+]] = arith.subi %[[V_12]], %c1{{.*}} : i64
  ! CHECK-DAG:     %[[V_23:[0-9]+]] = arith.addi %[[V_12]], %c1{{.*}} : i64
  ! CHECK:         %[[V_24:[0-9]+]] = arith.select %[[V_15]], %[[V_23]], %[[V_22]] : i64
  ! CHECK:         %[[V_25:[0-9]+]] = arith.bitcast %[[V_24]] : i64 to f64
  ! CHECK:         %[[V_26:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_25]]) <{bit = 516 : i32}> : (f64) -> i1
  ! CHECK:         fir.if %[[V_26]] {
  ! CHECK:           %[[V_28:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                              fir.call @feraiseexcept(%[[V_28]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         %[[V_27:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_25]]) <{bit = 144 : i32}> : (f64) -> i1
  ! CHECK:         fir.if %[[V_27]] {
  ! CHECK:           %[[V_28:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:                              fir.call @feraiseexcept(%[[V_28]]) fastmath<contract> : (i32) -> i32
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_25]] : f64
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_21]] : f64
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_19]] to %[[V_2]] : !fir.ref<f64>
  ! CHECK:     return
  ! CHECK:   }
subroutine nearest_test4(x, s)
  real(kind=8) :: x, s, res
  res = nearest(x, s)
end

! CHECK-KIND10-LABEL: c.func @_QPnearest_test5
  ! CHECK-KIND10:     %[[V_0:[0-9]+]] = fir.dummy_scope : !fir.dscope
  ! CHECK-KIND10:     %[[V_1:[0-9]+]] = fir.alloca f80 {bindc_name = "res", uniq_name = "_QFnearest_test5Eres"}
  ! CHECK-KIND10:     %[[V_2:[0-9]+]] = fir.declare %[[V_1]] {uniq_name = "_QFnearest_test5Eres"} : (!fir.ref<f80>) -> !fir.ref<f80>
  ! CHECK-KIND10:     %[[V_3:[0-9]+]] = fir.declare %arg1 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test5Es"} : (!fir.ref<f80>, !fir.dscope) -> !fir.ref<f80>
  ! CHECK-KIND10:     %[[V_4:[0-9]+]] = fir.declare %arg0 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test5Ex"} : (!fir.ref<f80>, !fir.dscope) -> !fir.ref<f80>
  ! CHECK-KIND10:     %[[V_5:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f80>
  ! CHECK-KIND10:     %[[V_6:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f80>
  ! CHECK-KIND10:     %[[V_7:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 3 : i32}> : (f80) -> i1
  ! CHECK-KIND10:     %[[V_8:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_6]]) <{bit = 96 : i32}> : (f80) -> i1
  ! CHECK-KIND10:     fir.if %[[V_8]] {
  ! CHECK-KIND10:       fir.call @_FortranAReportFatalUserError
  ! CHECK-KIND10:     }
  ! CHECK-KIND10:     %[[V_9:[0-9]+]] = arith.bitcast %[[V_6]] : f80 to i80
  ! CHECK-KIND10:     %[[V_10:[0-9]+]] = arith.shrui %[[V_9]], %c79{{.*}} : i80
  ! CHECK-KIND10:     %[[V_11:[0-9]+]] = arith.cmpi ne, %[[V_10]], %c1{{.*}} : i80
  ! CHECK-KIND10:     %[[V_12:[0-9]+]] = arith.bitcast %[[V_5]] : f80 to i80
  ! CHECK-KIND10:     %[[V_13:[0-9]+]] = arith.shrui %[[V_12]], %c79{{.*}} : i80
  ! CHECK-KIND10:     %[[V_14:[0-9]+]] = fir.convert %[[V_13]] : (i80) -> i1
  ! CHECK-KIND10:     %[[V_15:[0-9]+]] = arith.cmpi ne, %[[V_11]], %[[V_14]] : i1
  ! CHECK-KIND10:     %[[V_16:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 516 : i32}> : (f80) -> i1
  ! CHECK-KIND10:     %[[V_17:[0-9]+]] = arith.andi %[[V_16]], %[[V_15]] : i1
  ! CHECK-KIND10:     %[[V_18:[0-9]+]] = arith.ori %[[V_7]], %[[V_17]] : i1
  ! CHECK-KIND10:     %[[V_19:[0-9]+]] = fir.if %[[V_18]] -> (f80) {
  ! CHECK-KIND10:       fir.result %[[V_5]] : f80
  ! CHECK-KIND10:     } else {
  ! CHECK-KIND10:       %[[V_20:[0-9]+]] = arith.cmpf oeq, %[[V_5]], %cst{{[_0-9]*}} fastmath<contract> : f80
  ! CHECK-KIND10:       %[[V_21:[0-9]+]] = fir.if %[[V_20]] -> (f80) {
  ! CHECK-KIND10:         %[[V_22:[0-9]+]] = arith.select %[[V_11]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : f80
  ! CHECK-KIND10:         %[[V_23:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND10:                            fir.call @feraiseexcept(%[[V_23]]) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND10:         fir.result %[[V_22]] : f80
  ! CHECK-KIND10:       } else {
  ! CHECK-KIND10:         %[[V_22:[0-9]+]] = fir.call @_FortranANearest10(%[[V_5]], %[[V_11]]) fastmath<contract> : (f80, i1) -> f80
  ! CHECK-KIND10:         fir.result %[[V_22]] : f80
  ! CHECK-KIND10:       }
  ! CHECK-KIND10:       fir.result %[[V_21]] : f80
  ! CHECK-KIND10:     }
  ! CHECK-KIND10:     fir.store %[[V_19]] to %[[V_2]] : !fir.ref<f80>
  ! CHECK-KIND10:     return
  ! CHECK-KIND10:   }
subroutine nearest_test5(x, s)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(kind=kind10) :: x, s, res
  res = nearest(x, s)
end

! CHECK-KIND16-LABEL: c.func @_QPnearest_test6
  ! CHECK-KIND16:     %[[V_0:[0-9]+]] = fir.dummy_scope : !fir.dscope
  ! CHECK-KIND16:     %[[V_1:[0-9]+]] = fir.alloca f128 {bindc_name = "res", uniq_name = "_QFnearest_test6Eres"}
  ! CHECK-KIND16:     %[[V_2:[0-9]+]] = fir.declare %[[V_1]] {uniq_name = "_QFnearest_test6Eres"} : (!fir.ref<f128>) -> !fir.ref<f128>
  ! CHECK-KIND16:     %[[V_3:[0-9]+]] = fir.declare %arg1 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test6Es"} : (!fir.ref<f128>, !fir.dscope) -> !fir.ref<f128>
  ! CHECK-KIND16:     %[[V_4:[0-9]+]] = fir.declare %arg0 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test6Ex"} : (!fir.ref<f128>, !fir.dscope) -> !fir.ref<f128>
  ! CHECK-KIND16:     %[[V_5:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f128>
  ! CHECK-KIND16:     %[[V_6:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f128>
  ! CHECK-KIND16:     %[[V_7:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 3 : i32}> : (f128) -> i1
  ! CHECK-KIND16:     %[[V_8:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_6]]) <{bit = 96 : i32}> : (f128) -> i1
  ! CHECK-KIND16:     fir.if %[[V_8]] {
  ! CHECK-KIND16:       fir.call @_FortranAReportFatalUserError
  ! CHECK-KIND16:     }
  ! CHECK-KIND16:     %[[V_9:[0-9]+]] = arith.bitcast %[[V_6]] : f128 to i128
  ! CHECK-KIND16:     %[[V_10:[0-9]+]] = arith.shrui %[[V_9]], %c127{{.*}} : i128
  ! CHECK-KIND16:     %[[V_11:[0-9]+]] = arith.cmpi ne, %[[V_10]], %c1{{.*}} : i128
  ! CHECK-KIND16:     %[[V_12:[0-9]+]] = arith.bitcast %[[V_5]] : f128 to i128
  ! CHECK-KIND16:     %[[V_13:[0-9]+]] = arith.shrui %[[V_12]], %c127{{.*}} : i128
  ! CHECK-KIND16:     %[[V_14:[0-9]+]] = fir.convert %[[V_13]] : (i128) -> i1
  ! CHECK-KIND16:     %[[V_15:[0-9]+]] = arith.cmpi ne, %[[V_11]], %[[V_14]] : i1
  ! CHECK-KIND16:     %[[V_16:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 516 : i32}> : (f128) -> i1
  ! CHECK-KIND16:     %[[V_17:[0-9]+]] = arith.andi %[[V_16]], %[[V_15]] : i1
  ! CHECK-KIND16:     %[[V_18:[0-9]+]] = arith.ori %[[V_7]], %[[V_17]] : i1
  ! CHECK-KIND16:     %[[V_19:[0-9]+]] = fir.if %[[V_18]] -> (f128) {
  ! CHECK-KIND16:       fir.result %[[V_5]] : f128
  ! CHECK-KIND16:     } else {
  ! CHECK-KIND16:       %[[V_20:[0-9]+]] = arith.cmpf oeq, %[[V_5]], %cst{{[_0-9]*}} fastmath<contract> : f128
  ! CHECK-KIND16:       %[[V_21:[0-9]+]] = fir.if %[[V_20]] -> (f128) {
  ! CHECK-KIND16:         %[[V_22:[0-9]+]] = arith.select %[[V_11]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : f128
  ! CHECK-KIND16:         %[[V_23:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:                            fir.call @feraiseexcept(%[[V_23]]) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:         fir.result %[[V_22]] : f128
  ! CHECK-KIND16:       } else {
  ! CHECK-KIND16-DAG:     %[[V_22:[0-9]+]] = arith.subi %[[V_12]], %c1{{.*}} : i128
  ! CHECK-KIND16-DAG:     %[[V_23:[0-9]+]] = arith.addi %[[V_12]], %c1{{.*}} : i128
  ! CHECK-KIND16:         %[[V_24:[0-9]+]] = arith.select %[[V_15]], %[[V_23]], %[[V_22]] : i128
  ! CHECK-KIND16:         %[[V_25:[0-9]+]] = arith.bitcast %[[V_24]] : i128 to f128
  ! CHECK-KIND16:         %[[V_26:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_25]]) <{bit = 516 : i32}> : (f128) -> i1
  ! CHECK-KIND16:         fir.if %[[V_26]] {
  ! CHECK-KIND16:           %[[V_28:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:                              fir.call @feraiseexcept(%[[V_28]]) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:         }
  ! CHECK-KIND16:         %[[V_27:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_25]]) <{bit = 144 : i32}> : (f128) -> i1
  ! CHECK-KIND16:         fir.if %[[V_27]] {
  ! CHECK-KIND16:           %[[V_28:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:                              fir.call @feraiseexcept(%[[V_28]]) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:         }
  ! CHECK-KIND16:         fir.result %[[V_25]] : f128
  ! CHECK-KIND16:       }
  ! CHECK-KIND16:       fir.result %[[V_21]] : f128
  ! CHECK-KIND16:     }
  ! CHECK-KIND16:     fir.store %[[V_19]] to %[[V_2]] : !fir.ref<f128>
  ! CHECK-KIND16:     return
  ! CHECK-KIND16:   }
subroutine nearest_test6(x, s)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind=kind16) :: x, s, res
  res = nearest(x, s)
end

! CHECK-KIND16-LABEL: c.func @_QPnearest_test7
  ! CHECK-KIND16:     %[[V_0:[0-9]+]] = fir.dummy_scope : !fir.dscope
  ! CHECK-KIND16:     %[[V_1:[0-9]+]] = fir.alloca f128 {bindc_name = "res", uniq_name = "_QFnearest_test7Eres"}
  ! CHECK-KIND16:     %[[V_2:[0-9]+]] = fir.declare %[[V_1]] {uniq_name = "_QFnearest_test7Eres"} : (!fir.ref<f128>) -> !fir.ref<f128>
  ! CHECK-KIND16:     %[[V_3:[0-9]+]] = fir.declare %arg1 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test7Es"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  ! CHECK-KIND16:     %[[V_4:[0-9]+]] = fir.declare %arg0 dummy_scope %[[V_0]] {uniq_name = "_QFnearest_test7Ex"} : (!fir.ref<f128>, !fir.dscope) -> !fir.ref<f128>
  ! CHECK-KIND16:     %[[V_5:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f128>
  ! CHECK-KIND16:     %[[V_6:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f32>
  ! CHECK-KIND16:     %[[V_7:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 3 : i32}> : (f128) -> i1
  ! CHECK-KIND16:     %[[V_8:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_6]]) <{bit = 96 : i32}> : (f32) -> i1
  ! CHECK-KIND16:     fir.if %[[V_8]] {
  ! CHECK-KIND16:       fir.call @_FortranAReportFatalUserError
  ! CHECK-KIND16:     }
  ! CHECK-KIND16:     %[[V_9:[0-9]+]] = arith.bitcast %[[V_6]] : f32 to i32
  ! CHECK-KIND16:     %[[V_10:[0-9]+]] = arith.shrui %[[V_9]], %c31{{.*}} : i32
  ! CHECK-KIND16:     %[[V_11:[0-9]+]] = fir.convert %[[V_10]] : (i32) -> i128
  ! CHECK-KIND16:     %[[V_12:[0-9]+]] = arith.cmpi ne, %[[V_11]], %c1{{.*}} : i128
  ! CHECK-KIND16:     %[[V_13:[0-9]+]] = arith.bitcast %[[V_5]] : f128 to i128
  ! CHECK-KIND16:     %[[V_14:[0-9]+]] = arith.shrui %[[V_13]], %c127{{.*}} : i128
  ! CHECK-KIND16:     %[[V_15:[0-9]+]] = fir.convert %[[V_14]] : (i128) -> i1
  ! CHECK-KIND16:     %[[V_16:[0-9]+]] = arith.cmpi ne, %[[V_12]], %[[V_15]] : i1
  ! CHECK-KIND16:     %[[V_17:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_5]]) <{bit = 516 : i32}> : (f128) -> i1
  ! CHECK-KIND16:     %[[V_18:[0-9]+]] = arith.andi %[[V_17]], %[[V_16]] : i1
  ! CHECK-KIND16:     %[[V_19:[0-9]+]] = arith.ori %[[V_7]], %[[V_18]] : i1
  ! CHECK-KIND16:     %[[V_20:[0-9]+]] = fir.if %[[V_19]] -> (f128) {
  ! CHECK-KIND16:       fir.result %[[V_5]] : f128
  ! CHECK-KIND16:     } else {
  ! CHECK-KIND16:       %[[V_21:[0-9]+]] = arith.cmpf oeq, %[[V_5]], %cst{{[_0-9]*}} fastmath<contract> : f128
  ! CHECK-KIND16:       %[[V_22:[0-9]+]] = fir.if %[[V_21]] -> (f128) {
  ! CHECK-KIND16:         %[[V_23:[0-9]+]] = arith.select %[[V_12]], %cst{{[_0-9]*}}, %cst{{[_0-9]*}} : f128
  ! CHECK-KIND16:         %[[V_24:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:                            fir.call @feraiseexcept(%[[V_24]]) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:         fir.result %[[V_23]] : f128
  ! CHECK-KIND16:       } else {
  ! CHECK-KIND16-DAG:     %[[V_23:[0-9]+]] = arith.subi %[[V_13]], %c1{{.*}} : i128
  ! CHECK-KIND16-DAG:     %[[V_24:[0-9]+]] = arith.addi %[[V_13]], %c1{{.*}} : i128
  ! CHECK-KIND16:         %[[V_25:[0-9]+]] = arith.select %[[V_16]], %[[V_24]], %[[V_23]] : i128
  ! CHECK-KIND16:         %[[V_26:[0-9]+]] = arith.bitcast %[[V_25]] : i128 to f128
  ! CHECK-KIND16:         %[[V_27:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_26]]) <{bit = 516 : i32}> : (f128) -> i1
  ! CHECK-KIND16:         fir.if %[[V_27]] {
  ! CHECK-KIND16:           %[[V_29:[0-9]+]] = fir.call @_FortranAMapException(%c40{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:                              fir.call @feraiseexcept(%[[V_29]]) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:         }
  ! CHECK-KIND16:         %[[V_28:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_26]]) <{bit = 144 : i32}> : (f128) -> i1
  ! CHECK-KIND16:         fir.if %[[V_28]] {
  ! CHECK-KIND16:           %[[V_29:[0-9]+]] = fir.call @_FortranAMapException(%c48{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:                              fir.call @feraiseexcept(%[[V_29]]) fastmath<contract> : (i32) -> i32
  ! CHECK-KIND16:         }
  ! CHECK-KIND16:         fir.result %[[V_26]] : f128
  ! CHECK-KIND16:       }
  ! CHECK-KIND16:       fir.result %[[V_22]] : f128
  ! CHECK-KIND16:     }
  ! CHECK-KIND16:     fir.store %[[V_20]] to %[[V_2]] : !fir.ref<f128>
  ! CHECK-KIND16:     return
  ! CHECK-KIND16:   }
subroutine nearest_test7(x, s)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind=kind16) :: x, res
  real :: s
  res = nearest(x, s)
end

! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

module ieee_next_tests
  use ieee_arithmetic, only: ieee_value, ieee_negative_inf, ieee_positive_inf
  use ieee_arithmetic, only: ieee_next_after, ieee_next_down, ieee_next_up
  implicit none
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
contains

subroutine test1(r2, x2, x10)
  real(2)  ::  r2,  x2
  real(kind10) :: x10
  r2 = ieee_next_after(x2, x10)
end subroutine
!CHECK-KIND10-LABEL:   func.func @_QMieee_next_testsPtest1(
!CHECK-KIND10:           %[[VAL_12:.*]]:2 = hlfir.declare {{.*}}r2"
!CHECK-KIND10:           %[[VAL_13:.*]]:2 = hlfir.declare {{.*}}x10"
!CHECK-KIND10:           %[[VAL_14:.*]]:2 = hlfir.declare {{.*}}x2"
!CHECK-KIND10:           %[[VAL_15:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<f16>
!CHECK-KIND10:           %[[VAL_16:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<f80>
!CHECK-KIND10-DAG:       %[[VAL_17:.*]] = "llvm.intr.is.fpclass"(%[[VAL_16]]) <{bit = 3 : i32}> : (f80) -> i1
!CHECK-KIND10-DAG:       %[[VAL_18:.*]] = arith.constant 2 : i8
!CHECK-KIND10-DAG:       %[[VAL_19:.*]] = fir.address_of(@_FortranAIeeeValueTable_2) : !fir.ref<!fir.array<12xi16>>
!CHECK-KIND10-DAG:       %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_19]], %[[VAL_18]] : (!fir.ref<!fir.array<12xi16>>, i8) -> !fir.ref<i16>
!CHECK-KIND10-DAG:       %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i16>
!CHECK-KIND10-DAG:       %[[VAL_22:.*]] = arith.bitcast %[[VAL_21]] : i16 to f16
!CHECK-KIND10:           %[[VAL_23:.*]] = arith.select %[[VAL_17]], %[[VAL_22]], %[[VAL_15]] : f16
!CHECK-KIND10:           %[[VAL_24:.*]] = "llvm.intr.is.fpclass"(%[[VAL_23]]) <{bit = 3 : i32}> : (f16) -> i1
!CHECK-KIND10:           %[[VAL_25:.*]] = arith.constant 1 : i16
!CHECK-KIND10:           %[[VAL_26:.*]] = fir.convert %[[VAL_23]] : (f16) -> f32
!CHECK-KIND10:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (f32) -> f80
!CHECK-KIND10:           %[[VAL_28:.*]] = arith.cmpf oeq, %[[VAL_27]], %[[VAL_16]] fastmath<contract> : f80
!CHECK-KIND10:           %[[VAL_29:.*]] = arith.ori %[[VAL_24]], %[[VAL_28]] : i1
!CHECK-KIND10:           %[[VAL_30:.*]] = arith.cmpf olt, %[[VAL_27]], %[[VAL_16]] fastmath<contract> : f80
!CHECK-KIND10:           %[[VAL_31:.*]] = arith.bitcast %[[VAL_15]] : f16 to i16
!CHECK-KIND10:           %[[VAL_32:.*]] = arith.constant 15 : i16
!CHECK-KIND10:           %[[VAL_33:.*]] = arith.shrui %[[VAL_31]], %[[VAL_32]] : i16
!CHECK-KIND10:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i16) -> i1
!CHECK-KIND10:           %[[VAL_35:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_34]] : i1
!CHECK-KIND10:           %[[VAL_36:.*]] = "llvm.intr.is.fpclass"(%[[VAL_23]]) <{bit = 516 : i32}> : (f16) -> i1
!CHECK-KIND10:           %[[VAL_37:.*]] = arith.andi %[[VAL_36]], %[[VAL_35]] : i1
!CHECK-KIND10:           %[[VAL_38:.*]] = arith.ori %[[VAL_29]], %[[VAL_37]] : i1
!CHECK-KIND10:           %[[VAL_39:.*]] = fir.if %[[VAL_38]] -> (f16) {
!CHECK-KIND10:             fir.result %[[VAL_23]] : f16
!CHECK-KIND10:           } else {
!CHECK-KIND10:             %[[VAL_40:.*]] = arith.constant 0.000000e+00 : f16
!CHECK-KIND10:             %[[VAL_41:.*]] = arith.cmpf oeq, %[[VAL_23]], %[[VAL_40]] fastmath<contract> : f16
!CHECK-KIND10:             %[[VAL_42:.*]] = fir.if %[[VAL_41]] -> (f16) {
!CHECK-KIND10:               %[[VAL_43:.*]] = arith.bitcast %[[VAL_25]] : i16 to f16
!CHECK-KIND10:               %[[VAL_44:.*]] = arith.constant -32767 : i16
!CHECK-KIND10:               %[[VAL_45:.*]] = arith.bitcast %[[VAL_44]] : i16 to f16
!CHECK-KIND10:               %[[VAL_46:.*]] = arith.select %[[VAL_30]], %[[VAL_43]], %[[VAL_45]] : f16
!CHECK-KIND10:               %[[VAL_47:.*]] = arith.constant 48 : i32
!CHECK-KIND10:               %[[VAL_48:.*]] = fir.call @_FortranAMapException(%[[VAL_47]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               %[[VAL_49:.*]] = fir.call @feraiseexcept(%[[VAL_48]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               fir.result %[[VAL_46]] : f16
!CHECK-KIND10:             } else {
!CHECK-KIND10:               %[[VAL_50:.*]] = arith.bitcast %[[VAL_23]] : f16 to i16
!CHECK-KIND10-DAG:           %[[VAL_51:.*]] = arith.addi %[[VAL_50]], %[[VAL_25]] : i16
!CHECK-KIND10-DAG:           %[[VAL_52:.*]] = arith.subi %[[VAL_50]], %[[VAL_25]] : i16
!CHECK-KIND10:               %[[VAL_53:.*]] = arith.select %[[VAL_35]], %[[VAL_51]], %[[VAL_52]] : i16
!CHECK-KIND10:               %[[VAL_54:.*]] = arith.bitcast %[[VAL_53]] : i16 to f16
!CHECK-KIND10:               %[[VAL_55:.*]] = "llvm.intr.is.fpclass"(%[[VAL_54]]) <{bit = 516 : i32}> : (f16) -> i1
!CHECK-KIND10:               fir.if %[[VAL_55]] {
!CHECK-KIND10:                 %[[VAL_56:.*]] = arith.constant 40 : i32
!CHECK-KIND10:                 %[[VAL_57:.*]] = fir.call @_FortranAMapException(%[[VAL_56]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:                 %[[VAL_58:.*]] = fir.call @feraiseexcept(%[[VAL_57]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               }
!CHECK-KIND10:               %[[VAL_59:.*]] = "llvm.intr.is.fpclass"(%[[VAL_54]]) <{bit = 144 : i32}> : (f16) -> i1
!CHECK-KIND10:               fir.if %[[VAL_59]] {
!CHECK-KIND10:                 %[[VAL_60:.*]] = arith.constant 48 : i32
!CHECK-KIND10:                 %[[VAL_61:.*]] = fir.call @_FortranAMapException(%[[VAL_60]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:                 %[[VAL_62:.*]] = fir.call @feraiseexcept(%[[VAL_61]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               }
!CHECK-KIND10:               fir.result %[[VAL_54]] : f16
!CHECK-KIND10:             }
!CHECK-KIND10:             fir.result %[[VAL_42]] : f16
!CHECK-KIND10:           }
!CHECK-KIND10:           hlfir.assign %[[VAL_39]] to %[[VAL_12]]#0 : f16, !fir.ref<f16>
!CHECK-KIND10:           return
!CHECK-KIND10:         }

subroutine test2(r3, x3)
  real(3)  ::  r3,  x3
  r3 = ieee_next_up(x3)
end subroutine
!CHECK-LABEL:   func.func @_QMieee_next_testsPtest2(
!CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}r3"
!CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare {{.*}}x3"
!CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<bf16>
!CHECK:           %[[VAL_14:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 3 : i32}> : (bf16) -> i1
!CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i16
!CHECK:           %[[VAL_16:.*]] = arith.constant true
!CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_13]] : (bf16) -> f32
!CHECK:           %[[VAL_18:.*]] = arith.bitcast %[[VAL_17]] : f32 to i32
!CHECK:           %[[VAL_19:.*]] = arith.constant 31 : i32
!CHECK:           %[[VAL_20:.*]] = arith.shrui %[[VAL_18]], %[[VAL_19]] : i32
!CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i1
!CHECK:           %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_21]] : i1
!CHECK:           %[[VAL_23:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 516 : i32}> : (bf16) -> i1
!CHECK:           %[[VAL_24:.*]] = arith.andi %[[VAL_23]], %[[VAL_22]] : i1
!CHECK:           %[[VAL_25:.*]] = arith.ori %[[VAL_14]], %[[VAL_24]] : i1
!CHECK:           %[[VAL_26:.*]] = fir.if %[[VAL_25]] -> (bf16) {
!CHECK:             %[[VAL_27:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 1 : i32}> : (bf16) -> i1
!CHECK:             fir.if %[[VAL_27]] {
!CHECK:               %[[VAL_28:.*]] = arith.constant 1 : i32
!CHECK:               %[[VAL_29:.*]] = fir.call @_FortranAMapException(%[[VAL_28]]) fastmath<contract> : (i32) -> i32
!CHECK:               %[[VAL_30:.*]] = fir.call @feraiseexcept(%[[VAL_29]]) fastmath<contract> : (i32) -> i32
!CHECK:             }
!CHECK:             fir.result %[[VAL_13]] : bf16
!CHECK:           } else {
!CHECK:             %[[VAL_31:.*]] = arith.constant 0.000000e+00 : bf16
!CHECK:             %[[VAL_32:.*]] = arith.cmpf oeq, %[[VAL_13]], %[[VAL_31]] fastmath<contract> : bf16
!CHECK:             %[[VAL_33:.*]] = fir.if %[[VAL_32]] -> (bf16) {
!CHECK:               %[[VAL_34:.*]] = arith.bitcast %[[VAL_15]] : i16 to bf16
!CHECK:               %[[VAL_35:.*]] = arith.constant -32767 : i16
!CHECK:               %[[VAL_36:.*]] = arith.bitcast %[[VAL_35]] : i16 to bf16
!CHECK:               %[[VAL_37:.*]] = arith.select %[[VAL_16]], %[[VAL_34]], %[[VAL_36]] : bf16
!CHECK:               fir.result %[[VAL_37]] : bf16
!CHECK:             } else {
!CHECK:               %[[VAL_38:.*]] = arith.bitcast %[[VAL_13]] : bf16 to i16
!CHECK-DAG:           %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_15]] : i16
!CHECK-DAG:           %[[VAL_40:.*]] = arith.subi %[[VAL_38]], %[[VAL_15]] : i16
!CHECK:               %[[VAL_41:.*]] = arith.select %[[VAL_22]], %[[VAL_39]], %[[VAL_40]] : i16
!CHECK:               %[[VAL_42:.*]] = arith.bitcast %[[VAL_41]] : i16 to bf16
!CHECK:               fir.result %[[VAL_42]] : bf16
!CHECK:             }
!CHECK:             fir.result %[[VAL_33]] : bf16
!CHECK:           }
!CHECK:           hlfir.assign %[[VAL_26]] to %[[VAL_11]]#0 : bf16, !fir.ref<bf16>
!CHECK:           return
!CHECK:         }

subroutine test3(r4, x4)
  real(4)  ::  r4,  x4
  r4 = ieee_next_down(x4)
end subroutine
!CHECK-LABEL:   func.func @_QMieee_next_testsPtest3(
!CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}r4"
!CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare {{.*}}x4"
!CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<f32>
!CHECK:           %[[VAL_14:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 3 : i32}> : (f32) -> i1
!CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_16:.*]] = arith.constant false
!CHECK:           %[[VAL_17:.*]] = arith.bitcast %[[VAL_13]] : f32 to i32
!CHECK:           %[[VAL_18:.*]] = arith.constant 31 : i32
!CHECK:           %[[VAL_19:.*]] = arith.shrui %[[VAL_17]], %[[VAL_18]] : i32
!CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i1
!CHECK:           %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_20]] : i1
!CHECK:           %[[VAL_22:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 516 : i32}> : (f32) -> i1
!CHECK:           %[[VAL_23:.*]] = arith.andi %[[VAL_22]], %[[VAL_21]] : i1
!CHECK:           %[[VAL_24:.*]] = arith.ori %[[VAL_14]], %[[VAL_23]] : i1
!CHECK:           %[[VAL_25:.*]] = fir.if %[[VAL_24]] -> (f32) {
!CHECK:             %[[VAL_26:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 1 : i32}> : (f32) -> i1
!CHECK:             fir.if %[[VAL_26]] {
!CHECK:               %[[VAL_27:.*]] = arith.constant 1 : i32
!CHECK:               %[[VAL_28:.*]] = fir.call @_FortranAMapException(%[[VAL_27]]) fastmath<contract> : (i32) -> i32
!CHECK:               %[[VAL_29:.*]] = fir.call @feraiseexcept(%[[VAL_28]]) fastmath<contract> : (i32) -> i32
!CHECK:             }
!CHECK:             fir.result %[[VAL_13]] : f32
!CHECK:           } else {
!CHECK:             %[[VAL_30:.*]] = arith.constant 0.000000e+00 : f32
!CHECK:             %[[VAL_31:.*]] = arith.cmpf oeq, %[[VAL_13]], %[[VAL_30]] fastmath<contract> : f32
!CHECK:             %[[VAL_32:.*]] = fir.if %[[VAL_31]] -> (f32) {
!CHECK:               %[[VAL_33:.*]] = arith.bitcast %[[VAL_15]] : i32 to f32
!CHECK:               %[[VAL_34:.*]] = arith.constant -2147483647 : i32
!CHECK:               %[[VAL_35:.*]] = arith.bitcast %[[VAL_34]] : i32 to f32
!CHECK:               %[[VAL_36:.*]] = arith.select %[[VAL_16]], %[[VAL_33]], %[[VAL_35]] : f32
!CHECK:               fir.result %[[VAL_36]] : f32
!CHECK:             } else {
!CHECK:               %[[VAL_37:.*]] = arith.bitcast %[[VAL_13]] : f32 to i32
!CHECK-DAG:           %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_15]] : i32
!CHECK-DAG:           %[[VAL_39:.*]] = arith.subi %[[VAL_37]], %[[VAL_15]] : i32
!CHECK:               %[[VAL_40:.*]] = arith.select %[[VAL_21]], %[[VAL_38]], %[[VAL_39]] : i32
!CHECK:               %[[VAL_41:.*]] = arith.bitcast %[[VAL_40]] : i32 to f32
!CHECK:               fir.result %[[VAL_41]] : f32
!CHECK:             }
!CHECK:             fir.result %[[VAL_32]] : f32
!CHECK:           }
!CHECK:           hlfir.assign %[[VAL_25]] to %[[VAL_11]]#0 : f32, !fir.ref<f32>
!CHECK:           return
!CHECK:         }

subroutine test4(r8, x8, x2)
  real(2)  ::  x2
  real(8)  ::  r8,  x8
  r8 = ieee_next_after(x8, x2)
end subroutine
!CHECK-LABEL:   func.func @_QMieee_next_testsPtest4(
!CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare {{.*}}r8"
!CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare {{.*}}x2"
!CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare {{.*}}x8"
!CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<f64>
!CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<f16>
!CHECK-DAG:       %[[VAL_17:.*]] = "llvm.intr.is.fpclass"(%[[VAL_16]]) <{bit = 3 : i32}> : (f16) -> i1
!CHECK-DAG:       %[[VAL_18:.*]] = arith.constant 2 : i8
!CHECK-DAG:       %[[VAL_19:.*]] = fir.address_of(@_FortranAIeeeValueTable_8) : !fir.ref<!fir.array<12xi64>>
!CHECK-DAG:       %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_19]], %[[VAL_18]] : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
!CHECK-DAG:       %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i64>
!CHECK-DAG:       %[[VAL_22:.*]] = arith.bitcast %[[VAL_21]] : i64 to f64
!CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_17]], %[[VAL_22]], %[[VAL_15]] : f64
!CHECK:           %[[VAL_24:.*]] = "llvm.intr.is.fpclass"(%[[VAL_23]]) <{bit = 3 : i32}> : (f64) -> i1
!CHECK:           %[[VAL_25:.*]] = arith.constant 1 : i64
!CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_16]] : (f16) -> f32
!CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (f32) -> f64
!CHECK:           %[[VAL_28:.*]] = arith.cmpf oeq, %[[VAL_23]], %[[VAL_27]] fastmath<contract> : f64
!CHECK:           %[[VAL_29:.*]] = arith.ori %[[VAL_24]], %[[VAL_28]] : i1
!CHECK:           %[[VAL_30:.*]] = arith.cmpf olt, %[[VAL_23]], %[[VAL_27]] fastmath<contract> : f64
!CHECK:           %[[VAL_31:.*]] = arith.bitcast %[[VAL_15]] : f64 to i64
!CHECK:           %[[VAL_32:.*]] = arith.constant 63 : i64
!CHECK:           %[[VAL_33:.*]] = arith.shrui %[[VAL_31]], %[[VAL_32]] : i64
!CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i64) -> i1
!CHECK:           %[[VAL_35:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_34]] : i1
!CHECK:           %[[VAL_36:.*]] = "llvm.intr.is.fpclass"(%[[VAL_23]]) <{bit = 516 : i32}> : (f64) -> i1
!CHECK:           %[[VAL_37:.*]] = arith.andi %[[VAL_36]], %[[VAL_35]] : i1
!CHECK:           %[[VAL_38:.*]] = arith.ori %[[VAL_29]], %[[VAL_37]] : i1
!CHECK:           %[[VAL_39:.*]] = fir.if %[[VAL_38]] -> (f64) {
!CHECK:             fir.result %[[VAL_23]] : f64
!CHECK:           } else {
!CHECK:             %[[VAL_40:.*]] = arith.constant 0.000000e+00 : f64
!CHECK:             %[[VAL_41:.*]] = arith.cmpf oeq, %[[VAL_23]], %[[VAL_40]] fastmath<contract> : f64
!CHECK:             %[[VAL_42:.*]] = fir.if %[[VAL_41]] -> (f64) {
!CHECK:               %[[VAL_43:.*]] = arith.bitcast %[[VAL_25]] : i64 to f64
!CHECK:               %[[VAL_44:.*]] = arith.constant -9223372036854775807 : i64
!CHECK:               %[[VAL_45:.*]] = arith.bitcast %[[VAL_44]] : i64 to f64
!CHECK:               %[[VAL_46:.*]] = arith.select %[[VAL_30]], %[[VAL_43]], %[[VAL_45]] : f64
!CHECK:               %[[VAL_47:.*]] = arith.constant 48 : i32
!CHECK:               %[[VAL_48:.*]] = fir.call @_FortranAMapException(%[[VAL_47]]) fastmath<contract> : (i32) -> i32
!CHECK:               %[[VAL_49:.*]] = fir.call @feraiseexcept(%[[VAL_48]]) fastmath<contract> : (i32) -> i32
!CHECK:               fir.result %[[VAL_46]] : f64
!CHECK:             } else {
!CHECK:               %[[VAL_50:.*]] = arith.bitcast %[[VAL_23]] : f64 to i64
!CHECK-DAG:           %[[VAL_51:.*]] = arith.addi %[[VAL_50]], %[[VAL_25]] : i64
!CHECK-DAG:           %[[VAL_52:.*]] = arith.subi %[[VAL_50]], %[[VAL_25]] : i64
!CHECK:               %[[VAL_53:.*]] = arith.select %[[VAL_35]], %[[VAL_51]], %[[VAL_52]] : i64
!CHECK:               %[[VAL_54:.*]] = arith.bitcast %[[VAL_53]] : i64 to f64
!CHECK:               %[[VAL_55:.*]] = "llvm.intr.is.fpclass"(%[[VAL_54]]) <{bit = 516 : i32}> : (f64) -> i1
!CHECK:               fir.if %[[VAL_55]] {
!CHECK:                 %[[VAL_56:.*]] = arith.constant 40 : i32
!CHECK:                 %[[VAL_57:.*]] = fir.call @_FortranAMapException(%[[VAL_56]]) fastmath<contract> : (i32) -> i32
!CHECK:                 %[[VAL_58:.*]] = fir.call @feraiseexcept(%[[VAL_57]]) fastmath<contract> : (i32) -> i32
!CHECK:               }
!CHECK:               %[[VAL_59:.*]] = "llvm.intr.is.fpclass"(%[[VAL_54]]) <{bit = 144 : i32}> : (f64) -> i1
!CHECK:               fir.if %[[VAL_59]] {
!CHECK:                 %[[VAL_60:.*]] = arith.constant 48 : i32
!CHECK:                 %[[VAL_61:.*]] = fir.call @_FortranAMapException(%[[VAL_60]]) fastmath<contract> : (i32) -> i32
!CHECK:                 %[[VAL_62:.*]] = fir.call @feraiseexcept(%[[VAL_61]]) fastmath<contract> : (i32) -> i32
!CHECK:               }
!CHECK:               fir.result %[[VAL_54]] : f64
!CHECK:             }
!CHECK:             fir.result %[[VAL_42]] : f64
!CHECK:           }
!CHECK:           hlfir.assign %[[VAL_39]] to %[[VAL_12]]#0 : f64, !fir.ref<f64>
!CHECK:           return
!CHECK:         }

subroutine test5(r10, x10)
  real(kind10) :: x10, r10
  r10 = ieee_next_up(x10)
end subroutine
!CHECK-KIND10-LABEL:   func.func @_QMieee_next_testsPtest5(
!CHECK-KIND10:           %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}r10"
!CHECK-KIND10:           %[[VAL_12:.*]]:2 = hlfir.declare {{.*}}x10"
!CHECK-KIND10:           %[[VAL_13:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<f80>
!CHECK-KIND10:           %[[VAL_14:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 3 : i32}> : (f80) -> i1
!CHECK-KIND10:           %[[VAL_15:.*]] = arith.constant 1 : i80
!CHECK-KIND10:           %[[VAL_16:.*]] = arith.constant true
!CHECK-KIND10:           %[[VAL_17:.*]] = arith.bitcast %[[VAL_13]] : f80 to i80
!CHECK-KIND10:           %[[VAL_18:.*]] = arith.constant 79 : i80
!CHECK-KIND10:           %[[VAL_19:.*]] = arith.shrui %[[VAL_17]], %[[VAL_18]] : i80
!CHECK-KIND10:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i80) -> i1
!CHECK-KIND10:           %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_20]] : i1
!CHECK-KIND10:           %[[VAL_22:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 516 : i32}> : (f80) -> i1
!CHECK-KIND10:           %[[VAL_23:.*]] = arith.andi %[[VAL_22]], %[[VAL_21]] : i1
!CHECK-KIND10:           %[[VAL_24:.*]] = arith.ori %[[VAL_14]], %[[VAL_23]] : i1
!CHECK-KIND10:           %[[VAL_25:.*]] = fir.if %[[VAL_24]] -> (f80) {
!CHECK-KIND10:             %[[VAL_26:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 1 : i32}> : (f80) -> i1
!CHECK-KIND10:             fir.if %[[VAL_26]] {
!CHECK-KIND10:               %[[VAL_27:.*]] = arith.constant 1 : i32
!CHECK-KIND10:               %[[VAL_28:.*]] = fir.call @_FortranAMapException(%[[VAL_27]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               %[[VAL_29:.*]] = fir.call @feraiseexcept(%[[VAL_28]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:             }
!CHECK-KIND10:             fir.result %[[VAL_13]] : f80
!CHECK-KIND10:           } else {
!CHECK-KIND10:             %[[VAL_30:.*]] = arith.constant 0.000000e+00 : f80
!CHECK-KIND10:             %[[VAL_31:.*]] = arith.cmpf oeq, %[[VAL_13]], %[[VAL_30]] fastmath<contract> : f80
!CHECK-KIND10:             %[[VAL_32:.*]] = fir.if %[[VAL_31]] -> (f80) {
!CHECK-KIND10:               %[[VAL_33:.*]] = arith.bitcast %[[VAL_15]] : i80 to f80
!CHECK-KIND10:               %[[VAL_34:.*]] = arith.constant -604462909807314587353087 : i80
!CHECK-KIND10:               %[[VAL_35:.*]] = arith.bitcast %[[VAL_34]] : i80 to f80
!CHECK-KIND10:               %[[VAL_36:.*]] = arith.select %[[VAL_16]], %[[VAL_33]], %[[VAL_35]] : f80
!CHECK-KIND10:               fir.result %[[VAL_36]] : f80
!CHECK-KIND10:             } else {
!CHECK-KIND10:               %[[VAL_37:.*]] = arith.constant 63 : i32
!CHECK-KIND10:               %[[VAL_38:.*]] = fir.call @_FortranAMapException(%[[VAL_37]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               %[[VAL_39:.*]] = fir.call @fetestexcept(%[[VAL_38]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               %[[VAL_40:.*]] = fir.call @fedisableexcept(%[[VAL_38]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               %[[VAL_41:.*]] = fir.call @_FortranANearest10(%[[VAL_13]], %[[VAL_16]]) fastmath<contract> : (f80, i1) -> f80
!CHECK-KIND10:               %[[VAL_42:.*]] = fir.call @feclearexcept(%[[VAL_38]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               %[[VAL_43:.*]] = fir.call @feraiseexcept(%[[VAL_39]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               %[[VAL_44:.*]] = fir.call @feenableexcept(%[[VAL_40]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND10:               fir.result %[[VAL_41]] : f80
!CHECK-KIND10:             }
!CHECK-KIND10:             fir.result %[[VAL_32]] : f80
!CHECK-KIND10:           }
!CHECK-KIND10:           hlfir.assign %[[VAL_25]] to %[[VAL_11]]#0 : f80, !fir.ref<f80>
!CHECK-KIND10:           return
!CHECK-KIND10:         }

subroutine test6(r16, x16)
  real(kind16) :: r16, x16
  r16 = ieee_next_down(x16)
end subroutine
!CHECK-KIND16-LABEL:   func.func @_QMieee_next_testsPtest6(
!CHECK-KIND16:           %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}r16"
!CHECK-KIND16:           %[[VAL_12:.*]]:2 = hlfir.declare {{.*}}x16"
!CHECK-KIND16:           %[[VAL_13:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<f128>
!CHECK-KIND16:           %[[VAL_14:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 3 : i32}> : (f128) -> i1
!CHECK-KIND16:           %[[VAL_15:.*]] = arith.constant 1 : i128
!CHECK-KIND16:           %[[VAL_16:.*]] = arith.constant false
!CHECK-KIND16:           %[[VAL_17:.*]] = arith.bitcast %[[VAL_13]] : f128 to i128
!CHECK-KIND16:           %[[VAL_18:.*]] = arith.constant 127 : i128
!CHECK-KIND16:           %[[VAL_19:.*]] = arith.shrui %[[VAL_17]], %[[VAL_18]] : i128
!CHECK-KIND16:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i128) -> i1
!CHECK-KIND16:           %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_20]] : i1
!CHECK-KIND16:           %[[VAL_22:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 516 : i32}> : (f128) -> i1
!CHECK-KIND16:           %[[VAL_23:.*]] = arith.andi %[[VAL_22]], %[[VAL_21]] : i1
!CHECK-KIND16:           %[[VAL_24:.*]] = arith.ori %[[VAL_14]], %[[VAL_23]] : i1
!CHECK-KIND16:           %[[VAL_25:.*]] = fir.if %[[VAL_24]] -> (f128) {
!CHECK-KIND16:             %[[VAL_26:.*]] = "llvm.intr.is.fpclass"(%[[VAL_13]]) <{bit = 1 : i32}> : (f128) -> i1
!CHECK-KIND16:             fir.if %[[VAL_26]] {
!CHECK-KIND16:               %[[VAL_27:.*]] = arith.constant 1 : i32
!CHECK-KIND16:               %[[VAL_28:.*]] = fir.call @_FortranAMapException(%[[VAL_27]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND16:               %[[VAL_29:.*]] = fir.call @feraiseexcept(%[[VAL_28]]) fastmath<contract> : (i32) -> i32
!CHECK-KIND16:             }
!CHECK-KIND16:             fir.result %[[VAL_13]] : f128
!CHECK-KIND16:           } else {
!CHECK-KIND16:             %[[VAL_30:.*]] = arith.constant 0.000000e+00 : f128
!CHECK-KIND16:             %[[VAL_31:.*]] = arith.cmpf oeq, %[[VAL_13]], %[[VAL_30]] fastmath<contract> : f128
!CHECK-KIND16:             %[[VAL_32:.*]] = fir.if %[[VAL_31]] -> (f128) {
!CHECK-KIND16:               %[[VAL_33:.*]] = arith.bitcast %[[VAL_15]] : i128 to f128
!CHECK-KIND16:               %[[VAL_34:.*]] = arith.constant -170141183460469231731687303715884105727 : i128
!CHECK-KIND16:               %[[VAL_35:.*]] = arith.bitcast %[[VAL_34]] : i128 to f128
!CHECK-KIND16:               %[[VAL_36:.*]] = arith.select %[[VAL_16]], %[[VAL_33]], %[[VAL_35]] : f128
!CHECK-KIND16:               fir.result %[[VAL_36]] : f128
!CHECK-KIND16:             } else {
!CHECK-KIND16:               %[[VAL_37:.*]] = arith.bitcast %[[VAL_13]] : f128 to i128
!CHECK-KIND16-DAG:           %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_15]] : i128
!CHECK-KIND16-DAG:           %[[VAL_39:.*]] = arith.subi %[[VAL_37]], %[[VAL_15]] : i128
!CHECK-KIND16:               %[[VAL_40:.*]] = arith.select %[[VAL_21]], %[[VAL_38]], %[[VAL_39]] : i128
!CHECK-KIND16:               %[[VAL_41:.*]] = arith.bitcast %[[VAL_40]] : i128 to f128
!CHECK-KIND16:               fir.result %[[VAL_41]] : f128
!CHECK-KIND16:             }
!CHECK-KIND16:             fir.result %[[VAL_32]] : f128
!CHECK-KIND16:           }
!CHECK-KIND16:           hlfir.assign %[[VAL_25]] to %[[VAL_11]]#0 : f128, !fir.ref<f128>
!CHECK-KIND16:           return
!CHECK-KIND16:         }
end module

! Expected end-to-end output when both kind10 and kind16 enabled (not part of lit
! test, only provided for debug help):
!
! after:  FC00 -> FBFF = -.655E+5
! up:     FF7F -> FF7E = -.337E+39
! down:   80000000 -> 80000001 = -.1E-44
! after:  0000000000000000 -> 8000000000000001 = -.5E-323
! up:     7FFEFFFFFFFFFFFFFFFF -> 7FFF8000000000000000 = Inf
! down:   7FFF0000000000000000000000000000 -> 7FFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF = .1189731495357231765085759326628007E+4933
program p
  use ieee_next_tests
  real(2)  ::  r2,  x2
  real(3)  ::  r3,  x3 = -huge(x3)
  real(4)  ::  r4,  x4 = -0.
  real(8)  ::  r8,  x8 =  0.
  real(kind10) :: r10, x10 =  huge(x10)
  real(kind16) :: r16, x16

  x2  = ieee_value(x2, ieee_negative_inf)
  x16 = ieee_value(x2, ieee_positive_inf)
  call test1(r2, x2, x10)
  print "('after:  ', z4.4, ' -> ', z4.4, ' = ', g0)", x2, r2, r2
  call test2(r3, x3)
  print "('up:     ', z4.4, ' -> ', z4.4, ' = ', g0)", x3, r3, r3
  call test3(r4, x4)
  print "('down:   ', z8.8, ' -> ', z8.8, ' = ', g0)", x4, r4, r4
  call test4(r8, x8, x2)
  print "('after:  ', z16.16, ' -> ', z16.16, ' = ', g0)", x8, r8, r8
  call test5(r10, x10)
  print "('up:     ', z20.20, ' -> ', z20.20, ' = ', g0)", x10, r10, r10
  call test6(r16, x16)
  print "('down:   ', z32.32, ' -> ', z32.32, ' = ', g0)", x16, r16, r16
end

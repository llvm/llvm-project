! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

subroutine sub1(a)
  integer :: a
  a = 1
end

! CHECK-LABEL: func @_QPsub1(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32>
! CHECK:         %[[C1:.*]] = arith.constant 1 : i32
! CHECK:         fir.store %[[C1]] to %[[ARG0]] : !fir.ref<i32>

subroutine sub2(a, b)
  integer(4) :: a
  integer(8) :: b
  a = b
end

! CHECK-LABEL: func @_QPsub2(
! CHECK:         %[[A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}
! CHECK:         %[[B:.*]]: !fir.ref<i64> {fir.bindc_name = "b"}
! CHECK:         %[[B_VAL:.*]] = fir.load %arg1 : !fir.ref<i64>
! CHECK:         %[[B_CONV:.*]] = fir.convert %[[B_VAL]] : (i64) -> i32
! CHECK:         fir.store %[[B_CONV]] to %[[A]] : !fir.ref<i32>

integer function negi(a)
  integer :: a
  negi = -a
end 

! CHECK-LABEL: func @_QPnegi(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) -> i32 {
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32 {bindc_name = "negi", uniq_name = "_QFnegiEnegi"}
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK:         %[[C0:.*]] = arith.constant 0 : i32
! CHECK:         %[[NEG:.*]] = arith.subi %[[C0]], %[[A_VAL]] : i32
! CHECK:         fir.store %[[NEG]] to %[[FCTRES]] : !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

real function negr(a)
  real :: a
  negr = -a
end 

! CHECK-LABEL: func @_QPnegr(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<f32> {fir.bindc_name = "a"}) -> f32 {
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32 {bindc_name = "negr", uniq_name = "_QFnegrEnegr"}
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<f32>
! CHECK:         %[[NEG:.*]] = arith.negf %[[A_VAL]] : f32
! CHECK:         fir.store %[[NEG]] to %[[FCTRES]] : !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

complex function negc(a)
  complex :: a
  negc = -a
end 

! CHECK-LABEL: func @_QPnegc(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "a"}) -> !fir.complex<4> {
! CHECK:         %[[FCTRES:.*]] = fir.alloca !fir.complex<4> {bindc_name = "negc", uniq_name = "_QFnegcEnegc"}
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[NEG:.*]] = fir.negc %[[A_VAL]] : !fir.complex<4>
! CHECK:         fir.store %[[NEG]] to %[[FCTRES]] : !fir.ref<!fir.complex<4>>

integer function addi(a, b)
  integer :: a, b
  addi = a + b
end

! CHECK-LABEL: func @_QPaddi(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
! CHECK:         %[[ADD:.*]] = arith.addi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK:         fir.store %[[ADD]] to %[[FCTRES]] : !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

integer function subi(a, b)
  integer :: a, b
  subi = a - b
end

! CHECK-LABEL: func @_QPsubi(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
! CHECK:         %[[SUB:.*]] = arith.subi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK:         fir.store %[[SUB]] to %[[FCTRES]] : !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

integer function muli(a, b)
  integer :: a, b
  muli = a * b
end

! CHECK-LABEL: func @_QPmuli(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
! CHECK:         %[[MUL:.*]] = arith.muli %[[A_VAL]], %[[B_VAL]] : i32
! CHECK:         fir.store %[[MUL]] to %[[FCTRES]] : !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

integer function divi(a, b)
  integer :: a, b
  divi = a / b
end

! CHECK-LABEL: func @_QPdivi(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<i32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<i32>
! CHECK:         %[[DIV:.*]] = arith.divsi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK:         fir.store %[[DIV]] to %[[FCTRES]] : !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

real function addf(a, b)
  real :: a, b
  addf = a + b
end

! CHECK-LABEL: func @_QPaddf(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<f32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<f32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<f32>
! CHECK:         %[[ADD:.*]] = arith.addf %[[A_VAL]], %[[B_VAL]] : f32
! CHECK:         fir.store %[[ADD]] to %[[FCTRES]] : !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

real function subf(a, b)
  real :: a, b
  subf = a - b
end

! CHECK-LABEL: func @_QPsubf(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<f32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<f32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<f32>
! CHECK:         %[[SUB:.*]] = arith.subf %[[A_VAL]], %[[B_VAL]] : f32
! CHECK:         fir.store %[[SUB]] to %[[FCTRES]] : !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

real function mulf(a, b)
  real :: a, b
  mulf = a * b
end

! CHECK-LABEL: func @_QPmulf(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<f32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<f32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<f32>
! CHECK:         %[[MUL:.*]] = arith.mulf %[[A_VAL]], %[[B_VAL]] : f32
! CHECK:         fir.store %[[MUL]] to %[[FCTRES]] : !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

real function divf(a, b)
  real :: a, b
  divf = a / b
end

! CHECK-LABEL: func @_QPdivf(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<f32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<f32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<f32>
! CHECK:         %[[DIV:.*]] = arith.divf %[[A_VAL]], %[[B_VAL]] : f32
! CHECK:         fir.store %[[DIV]] to %[[FCTRES]] : !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

complex function addc(a, b)
  complex :: a, b
  addc = a + b
end

! CHECK-LABEL: func @_QPaddc(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca !fir.complex<4>
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[ADD:.*]] = fir.addc %[[A_VAL]], %[[B_VAL]] : !fir.complex<4>
! CHECK:         fir.store %[[ADD]] to %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         return %[[RET]] : !fir.complex<4>

complex function subc(a, b)
  complex :: a, b
  subc = a - b
end

! CHECK-LABEL: func @_QPsubc(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca !fir.complex<4>
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[SUB:.*]] = fir.subc %[[A_VAL]], %[[B_VAL]] : !fir.complex<4>
! CHECK:         fir.store %[[SUB]] to %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         return %[[RET]] : !fir.complex<4>

complex function mulc(a, b)
  complex :: a, b
  mulc = a * b
end

! CHECK-LABEL: func @_QPmulc(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca !fir.complex<4>
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[MUL:.*]] = fir.mulc %[[A_VAL]], %[[B_VAL]] : !fir.complex<4>
! CHECK:         fir.store %[[MUL]] to %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         return %[[RET]] : !fir.complex<4>

complex function divc(a, b)
  complex :: a, b
  divc = a / b
end

! CHECK-LABEL: func @_QPdivc(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[B:.*]]: !fir.ref<!fir.complex<4>> {fir.bindc_name = "b"}
! CHECK:         %[[FCTRES:.*]] = fir.alloca !fir.complex<4>
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[DIV:.*]] = fir.divc %[[A_VAL]], %[[B_VAL]] : !fir.complex<4>
! CHECK:         fir.store %[[DIV]] to %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES]] : !fir.ref<!fir.complex<4>>
! CHECK:         return %[[RET]] : !fir.complex<4>

subroutine real_constant()
  real(2) :: a
  real(4) :: b
  real(8) :: c
  real(10) :: d
  real(16) :: e
  a = 2.0_2
  b = 4.0_4
  c = 8.0_8
  d = 10.0_10
  e = 16.0_16
end

! CHECK: %[[A:.*]] = fir.alloca f16
! CHECK: %[[B:.*]] = fir.alloca f32
! CHECK: %[[C:.*]] = fir.alloca f64
! CHECK: %[[D:.*]] = fir.alloca f80
! CHECK: %[[E:.*]] = fir.alloca f128
! CHECK: %[[C2:.*]] = arith.constant 2.000000e+00 : f16
! CHECK: fir.store %[[C2]] to %[[A]] : !fir.ref<f16>
! CHECK: %[[C4:.*]] = arith.constant 4.000000e+00 : f32
! CHECK: fir.store %[[C4]] to %[[B]] : !fir.ref<f32>
! CHECK: %[[C8:.*]] = arith.constant 8.000000e+00 : f64
! CHECK: fir.store %[[C8]] to %[[C]] : !fir.ref<f64>
! CHECK: %[[C10:.*]] = arith.constant 1.000000e+01 : f80
! CHECK: fir.store %[[C10]] to %[[D]] : !fir.ref<f80>
! CHECK: %[[C16:.*]] = arith.constant 1.600000e+01 : f128
! CHECK: fir.store %[[C16]] to %[[E]] : !fir.ref<f128>

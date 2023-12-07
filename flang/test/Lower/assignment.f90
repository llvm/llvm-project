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
! CHECK:         %[[NEG:.*]] = arith.negf %[[A_VAL]] {{.*}}: f32
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
! CHECK:         %[[ADD:.*]] = arith.addf %[[A_VAL]], %[[B_VAL]] {{.*}}: f32
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
! CHECK:         %[[SUB:.*]] = arith.subf %[[A_VAL]], %[[B_VAL]] {{.*}}: f32
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
! CHECK:         %[[MUL:.*]] = arith.mulf %[[A_VAL]], %[[B_VAL]] {{.*}}: f32
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
! CHECK:         %[[DIV:.*]] = arith.divf %[[A_VAL]], %[[B_VAL]] {{.*}}: f32
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
! CHECK:         %[[A_REAL:.*]] = fir.extract_value %[[A_VAL]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK:         %[[A_IMAG:.*]] = fir.extract_value %[[A_VAL]], [1 : index] : (!fir.complex<4>) -> f32
! CHECK:         %[[B_REAL:.*]] = fir.extract_value %[[B_VAL]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK:         %[[B_IMAG:.*]] = fir.extract_value %[[B_VAL]], [1 : index] : (!fir.complex<4>) -> f32
! CHECK:         %[[DIV:.*]] = fir.call @__divsc3(%[[A_REAL]], %[[A_IMAG]], %[[B_REAL]], %[[B_IMAG]]) fastmath<contract> : (f32, f32, f32, f32) -> !fir.complex<4>
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

subroutine complex_constant()
  complex(4) :: a
  a = (0, 1)
end

! CHECK-LABEL: func @_QPcomplex_constant()
! CHECK:         %[[A:.*]] = fir.alloca !fir.complex<4> {bindc_name = "a", uniq_name = "_QFcomplex_constantEa"}
! CHECK:         %[[C0:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:         %[[C1:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:         %[[UNDEF:.*]] = fir.undefined !fir.complex<4>
! CHECK:         %[[INS0:.*]] = fir.insert_value %[[UNDEF]], %[[C0]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:         %[[INS1:.*]] = fir.insert_value %[[INS0]], %[[C1]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:         fir.store %[[INS1]] to %[[A]] : !fir.ref<!fir.complex<4>>

subroutine sub1_arr(a)
  integer :: a(10)
  a(2) = 10
end

! CHECK-LABEL: func @_QPsub1_arr(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"})
! CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : i32
! CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : i64
! CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i64
! CHECK:         %[[ZERO_BASED_INDEX:.*]] = arith.subi %[[C2]], %[[C1]] : i64
! CHECK:         %[[COORD:.*]] = fir.coordinate_of %[[A]], %[[ZERO_BASED_INDEX]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:         fir.store %[[C10]] to %[[COORD]] : !fir.ref<i32>
! CHECK:         return

subroutine sub2_arr(a)
  integer :: a(10)
  a = 10
end

! CHECK-LABEL: func @_QPsub2_arr(
! CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"})
! CHECK-DAG:     %[[C10_0:.*]] = arith.constant 10 : index
! CHECK:         %[[SHAPE:.*]] = fir.shape %[[C10_0]] : (index) -> !fir.shape<1>
! CHECK:         %[[LOAD:.*]] = fir.array_load %[[A]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK-DAG:     %[[C10_1:.*]] = arith.constant 10 : i32
! CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
! CHECK-DAG:     %[[UB:.*]] = arith.subi %[[C10_0]], %c1 : index
! CHECK:         %[[DO_RES:.*]] = fir.do_loop %[[ARG1:.*]] = %[[C0]] to %[[UB]] step %[[C1]] unordered iter_args(%[[ARG2:.*]] = %[[LOAD]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[RES:.*]] = fir.array_update %[[ARG2]], %[[C10_1]], %[[ARG1]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[RES]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[LOAD]], %[[DO_RES]] to %[[A]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:         return

! RUN: %flang_fc1 %s -o "-" -emit-hlfir -cpp | FileCheck %s --check-prefixes=CHECK,%if flang-supports-f128-math %{F128%} %else %{F64%}%if target=x86_64-unknown-linux{{.*}} %{,CHECK-X86-64%}

subroutine sub1(a)
  integer :: a
  a = 1
end

! CHECK-LABEL: func @_QPsub1(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32>
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[C1:.*]] = arith.constant 1 : i32
! CHECK:         hlfir.assign %[[C1]] to %[[A]]#0 : i32, !fir.ref<i32>

subroutine sub2(a, b)
  integer(4) :: a
  integer(8) :: b
  a = b
end

! CHECK-LABEL: func @_QPsub2(
! CHECK:         %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}
! CHECK:         %[[ARG1:.*]]: !fir.ref<i64> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i64>
! CHECK:         %[[B_CONV:.*]] = fir.convert %[[B_VAL]] : (i64) -> i32
! CHECK:         hlfir.assign %[[B_CONV]] to %[[A]]#0 : i32, !fir.ref<i32>

integer function negi(a)
  integer :: a
  negi = -a
end

! CHECK-LABEL: func @_QPnegi(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) -> i32 {
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32 {bindc_name = "negi", uniq_name = "_QFnegiEnegi"}
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:         %[[C0:.*]] = arith.constant 0 : i32
! CHECK:         %[[NEG:.*]] = arith.subi %[[C0]], %[[A_VAL]] : i32
! CHECK:         hlfir.assign %[[NEG]] to %[[FCTRES_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

real function negr(a)
  real :: a
  negr = -a
end

! CHECK-LABEL: func @_QPnegr(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "a"}) -> f32 {
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32 {bindc_name = "negr", uniq_name = "_QFnegrEnegr"}
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<f32>
! CHECK:         %[[NEG:.*]] = arith.negf %[[A_VAL]] {{.*}}: f32
! CHECK:         hlfir.assign %[[NEG]] to %[[FCTRES_DECL]]#0 : f32, !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

complex function negc(a)
  complex :: a
  negc = -a
end

! CHECK-LABEL: func @_QPnegc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "a"}) -> complex<f32> {
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca complex<f32> {bindc_name = "negc", uniq_name = "_QFnegcEnegc"}
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[NEG:.*]] = fir.negc %[[A_VAL]] : complex<f32>
! CHECK:         hlfir.assign %[[NEG]] to %[[FCTRES_DECL]]#0 : complex<f32>, !fir.ref<complex<f32>>

integer function addi(a, b)
  integer :: a, b
  addi = a + b
end

! CHECK-LABEL: func @_QPaddi(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
! CHECK:         %[[ADD:.*]] = arith.addi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK:         hlfir.assign %[[ADD]] to %[[FCTRES_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

integer function subi(a, b)
  integer :: a, b
  subi = a - b
end

! CHECK-LABEL: func @_QPsubi(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
! CHECK:         %[[SUB:.*]] = arith.subi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK:         hlfir.assign %[[SUB]] to %[[FCTRES_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

integer function muli(a, b)
  integer :: a, b
  muli = a * b
end

! CHECK-LABEL: func @_QPmuli(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
! CHECK:         %[[MUL:.*]] = arith.muli %[[A_VAL]], %[[B_VAL]] : i32
! CHECK:         hlfir.assign %[[MUL]] to %[[FCTRES_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

integer function divi(a, b)
  integer :: a, b
  divi = a / b
end

! CHECK-LABEL: func @_QPdivi(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca i32
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<i32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<i32>
! CHECK:         %[[DIV:.*]] = arith.divsi %[[A_VAL]], %[[B_VAL]] : i32
! CHECK:         hlfir.assign %[[DIV]] to %[[FCTRES_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<i32>
! CHECK:         return %[[RET]] : i32

real function addf(a, b)
  real :: a, b
  addf = a + b
end

! CHECK-LABEL: func @_QPaddf(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<f32> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<f32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<f32>
! CHECK:         %[[ADD:.*]] = arith.addf %[[A_VAL]], %[[B_VAL]] {{.*}}: f32
! CHECK:         hlfir.assign %[[ADD]] to %[[FCTRES_DECL]]#0 : f32, !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

real function subf(a, b)
  real :: a, b
  subf = a - b
end

! CHECK-LABEL: func @_QPsubf(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<f32> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<f32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<f32>
! CHECK:         %[[SUB:.*]] = arith.subf %[[A_VAL]], %[[B_VAL]] {{.*}}: f32
! CHECK:         hlfir.assign %[[SUB]] to %[[FCTRES_DECL]]#0 : f32, !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

real function mulf(a, b)
  real :: a, b
  mulf = a * b
end

! CHECK-LABEL: func @_QPmulf(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<f32> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<f32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<f32>
! CHECK:         %[[MUL:.*]] = arith.mulf %[[A_VAL]], %[[B_VAL]] {{.*}}: f32
! CHECK:         hlfir.assign %[[MUL]] to %[[FCTRES_DECL]]#0 : f32, !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

real function divf(a, b)
  real :: a, b
  divf = a / b
end

! CHECK-LABEL: func @_QPdivf(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<f32> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca f32
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<f32>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<f32>
! CHECK:         %[[DIV:.*]] = arith.divf %[[A_VAL]], %[[B_VAL]] {{.*}}: f32
! CHECK:         hlfir.assign %[[DIV]] to %[[FCTRES_DECL]]#0 : f32, !fir.ref<f32>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<f32>
! CHECK:         return %[[RET]] : f32

complex function addc(a, b)
  complex :: a, b
  addc = a + b
end

! CHECK-LABEL: func @_QPaddc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca complex<f32>
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[ADD:.*]] = fir.addc %[[A_VAL]], %[[B_VAL]] {fastmath = #arith.fastmath<contract>} : complex<f32>
! CHECK:         hlfir.assign %[[ADD]] to %[[FCTRES_DECL]]#0 : complex<f32>, !fir.ref<complex<f32>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<complex<f32>>
! CHECK:         return %[[RET]] : complex<f32>

complex function subc(a, b)
  complex :: a, b
  subc = a - b
end

! CHECK-LABEL: func @_QPsubc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca complex<f32>
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[SUB:.*]] = fir.subc %[[A_VAL]], %[[B_VAL]] {fastmath = #arith.fastmath<contract>} : complex<f32>
! CHECK:         hlfir.assign %[[SUB]] to %[[FCTRES_DECL]]#0 : complex<f32>, !fir.ref<complex<f32>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<complex<f32>>
! CHECK:         return %[[RET]] : complex<f32>

complex function mulc(a, b)
  complex :: a, b
  mulc = a * b
end

! CHECK-LABEL: func @_QPmulc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca complex<f32>
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[MUL:.*]] = fir.mulc %[[A_VAL]], %[[B_VAL]] {fastmath = #arith.fastmath<contract>} : complex<f32>
! CHECK:         hlfir.assign %[[MUL]] to %[[FCTRES_DECL]]#0 : complex<f32>, !fir.ref<complex<f32>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<complex<f32>>
! CHECK:         return %[[RET]] : complex<f32>

complex function divc(a, b)
  complex :: a, b
  divc = a / b
end

! CHECK-LABEL: func @_QPdivc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "a"},
! CHECK-SAME:    %[[ARG1:.*]]: !fir.ref<complex<f32>> {fir.bindc_name = "b"}
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:         %[[FCTRES:.*]] = fir.alloca complex<f32>
! CHECK:         %[[FCTRES_DECL:.*]]:2 = hlfir.declare %[[FCTRES]]
! CHECK:         %[[A_VAL:.*]] = fir.load %[[A]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[B_VAL:.*]] = fir.load %[[B]]#0 : !fir.ref<complex<f32>>
! CHECK:         %[[A_REAL:.*]] = fir.extract_value %[[A_VAL]], [0 : index] : (complex<f32>) -> f32
! CHECK:         %[[A_IMAG:.*]] = fir.extract_value %[[A_VAL]], [1 : index] : (complex<f32>) -> f32
! CHECK:         %[[B_REAL:.*]] = fir.extract_value %[[B_VAL]], [0 : index] : (complex<f32>) -> f32
! CHECK:         %[[B_IMAG:.*]] = fir.extract_value %[[B_VAL]], [1 : index] : (complex<f32>) -> f32
! CHECK:         %[[DIV:.*]] = fir.call @__divsc3(%[[A_REAL]], %[[A_IMAG]], %[[B_REAL]], %[[B_IMAG]]) fastmath<contract> : (f32, f32, f32, f32) -> complex<f32>
! CHECK:         hlfir.assign %[[DIV]] to %[[FCTRES_DECL]]#0 : complex<f32>, !fir.ref<complex<f32>>
! CHECK:         %[[RET:.*]] = fir.load %[[FCTRES_DECL]]#0 : !fir.ref<complex<f32>>
! CHECK:         return %[[RET]] : complex<f32>

subroutine real_constant()
  integer, parameter :: rk = merge(16, 8, selected_real_kind(33, 4931)==16)
  real(2) :: a
  real(4) :: b
  real(8) :: c
#if __x86_64__
  real(10) :: d
#endif
  real(rk) :: e
  a = 2.0_2
  b = 4.0_4
  c = 8.0_8
#if __x86_64__
  d = 10.0_10
#endif
  e = 16.0_rk
end

! CHECK: %[[A:.*]] = fir.alloca f16
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[B:.*]] = fir.alloca f32
! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca f64
! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]]
! CHECK-X86-64: %[[D:.*]] = fir.alloca f80
! CHECK-X86-64: %[[D_DECL:.*]]:2 = hlfir.declare %[[D]]
! F128: %[[E:.*]] = fir.alloca f128
! F64: %[[E:.*]] = fir.alloca f64
! F128: %[[E_DECL:.*]]:2 = hlfir.declare %[[E]]
! F64: %[[E_DECL:.*]]:2 = hlfir.declare %[[E]]
! CHECK: %[[C2:.*]] = arith.constant 2.000000e+00 : f16
! CHECK: hlfir.assign %[[C2]] to %[[A_DECL]]#0 : f16, !fir.ref<f16>
! CHECK: %[[C4:.*]] = arith.constant 4.000000e+00 : f32
! CHECK: hlfir.assign %[[C4]] to %[[B_DECL]]#0 : f32, !fir.ref<f32>
! CHECK: %[[C8:.*]] = arith.constant 8.000000e+00 : f64
! CHECK: hlfir.assign %[[C8]] to %[[C_DECL]]#0 : f64, !fir.ref<f64>
! CHECK-X86-64: %[[C10:.*]] = arith.constant 1.000000e+01 : f80
! CHECK-X86-64: hlfir.assign %[[C10]] to %[[D_DECL]]#0 : f80, !fir.ref<f80>
! F128: %[[C16:.*]] = arith.constant 1.600000e+01 : f128
! F64: %[[C16:.*]] = arith.constant 1.600000e+01 : f64
! F128: hlfir.assign %[[C16]] to %[[E_DECL]]#0 : f128, !fir.ref<f128>
! F64: hlfir.assign %[[C16]] to %[[E_DECL]]#0 : f64, !fir.ref<f64>

subroutine complex_constant()
  complex(4) :: a
  a = (0, 1)
end

! CHECK-LABEL: func @_QPcomplex_constant()
! CHECK:         %[[A:.*]] = fir.alloca complex<f32> {bindc_name = "a", uniq_name = "_QFcomplex_constantEa"}
! CHECK:         %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]
! CHECK:         %[[C0:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:         %[[C1:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:         %[[UNDEF:.*]] = fir.undefined complex<f32>
! CHECK:         %[[INS0:.*]] = fir.insert_value %[[UNDEF]], %[[C0]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:         %[[INS1:.*]] = fir.insert_value %[[INS0]], %[[C1]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:         hlfir.assign %[[INS1]] to %[[A_DECL]]#0 : complex<f32>, !fir.ref<complex<f32>>

subroutine sub1_arr(a)
  integer :: a(10)
  a(2) = 10
end

! CHECK-LABEL: func @_QPsub1_arr(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"})
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]({{.*}}) {{.*}} {uniq_name = "_QFsub1_arrEa"}
! CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : i32
! CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
! CHECK:         %[[ELEM:.*]] = hlfir.designate %[[A]]#0 (%[[C2]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:         hlfir.assign %[[C10]] to %[[ELEM]] : i32, !fir.ref<i32>
! CHECK:         return

subroutine sub2_arr(a)
  integer :: a(10)
  a = 10
end

! CHECK-LABEL: func @_QPsub2_arr(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "a"})
! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[ARG0]]({{.*}}) {{.*}} {uniq_name = "_QFsub2_arrEa"}
! CHECK:         %[[C10:.*]] = arith.constant 10 : i32
! CHECK:         hlfir.assign %[[C10]] to %[[A]]#0 : i32, !fir.ref<!fir.array<10xi32>>
! CHECK:         return

! RUN: split-file %s %t

//--- abs.f90
! RUN: bbc -emit-fir %t/abs.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/abs.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/abs.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/abs.f90
! RUN: bbc -emit-fir %t/abs.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/abs.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/abs.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/abs.f90
! RUN: bbc -emit-fir %t/abs.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/abs.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/abs.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/abs.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = abs(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.abs {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.abs {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @fabsf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = abs(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.abs {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.abs {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @fabs({{%[A-Za-z0-9._]+}}) : (f64) -> f64

function test_real16(x)
  real(16) :: x, test_real16
  test_real16 = abs(x)
end function
! ALL-LABEL: @_QPtest_real16
! FAST: {{%[A-Za-z0-9._]+}} = math.abs {{%[A-Za-z0-9._]+}} : f128
! RELAXED: {{%[A-Za-z0-9._]+}} = math.abs {{%[A-Za-z0-9._]+}} : f128
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.fabs.f128({{%[A-Za-z0-9._]+}}) : (f128) -> f128

function test_complex4(c)
  complex(4) :: c, test_complex4
  test_complex4 = abs(c)
end function

! ALL-LABEL: @_QPtest_complex4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @hypotf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

function test_complex8(c)
  complex(8) :: c, test_complex8
  test_complex8 = abs(c)
end function

! ALL-LABEL: @_QPtest_complex8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @hypot({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

//--- aint.f90
! RUN: bbc -emit-fir %t/aint.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %t/aint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/aint.f90 -o - | FileCheck --check-prefixes=ALL %t/aint.f90
! RUN: bbc -emit-fir %t/aint.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %t/aint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/aint.f90 -o - | FileCheck --check-prefixes=ALL %t/aint.f90
! RUN: bbc -emit-fir %t/aint.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %t/aint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/aint.f90 -o - | FileCheck --check-prefixes=ALL %t/aint.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = aint(x)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.trunc.f32({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = aint(x)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.trunc.f64({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- anint.f90
! RUN: bbc -emit-fir %t/anint.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/anint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/anint.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/anint.f90
! RUN: bbc -emit-fir %t/anint.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/anint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/anint.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/anint.f90
! RUN: bbc -emit-fir %t/anint.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/anint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/anint.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/anint.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = anint(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = "llvm.intr.round"({{%[A-Za-z0-9._]+}}) : (f32) -> f32
! RELAXED: {{%[A-Za-z0-9._]+}} = "llvm.intr.round"({{%[A-Za-z0-9._]+}}) : (f32) -> f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.round.f32({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = anint(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = "llvm.intr.round"({{%[A-Za-z0-9._]+}}) : (f64) -> f64
! RELAXED: {{%[A-Za-z0-9._]+}} = "llvm.intr.round"({{%[A-Za-z0-9._]+}}) : (f64) -> f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.round.f64({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- atan.f90
! RUN: bbc -emit-fir %t/atan.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/atan.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/atan.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/atan.f90
! RUN: bbc -emit-fir %t/atan.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/atan.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/atan.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/atan.f90
! RUN: bbc -emit-fir %t/atan.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/atan.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/atan.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/atan.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = atan(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.atan {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.atan {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @atanf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = atan(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.atan {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.atan {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @atan({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- atan2.f90
! RUN: bbc -emit-fir %t/atan2.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/atan2.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/atan2.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/atan2.f90
! RUN: bbc -emit-fir %t/atan2.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/atan2.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/atan2.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/atan2.f90
! RUN: bbc -emit-fir %t/atan2.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/atan2.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/atan2.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/atan2.f90

function test_real4(x, y)
  real :: x, y, test_real4
  test_real4 = atan2(x, y)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.atan2 {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.atan2 {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @atan2f({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

function test_real8(x, y)
  real(8) :: x, y, test_real8
  test_real8 = atan2(x, y)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.atan2 {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.atan2 {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @atan2({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

//--- ceiling.f90
! RUN: bbc -emit-fir %t/ceiling.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/ceiling.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/ceiling.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/ceiling.f90
! RUN: bbc -emit-fir %t/ceiling.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/ceiling.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/ceiling.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/ceiling.f90
! RUN: bbc -emit-fir %t/ceiling.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/ceiling.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/ceiling.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/ceiling.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = ceiling(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.ceil {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.ceil {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @ceilf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = ceiling(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.ceil {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.ceil {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @ceil({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- cos.f90
! RUN: bbc -emit-fir %t/cos.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/cos.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/cos.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/cos.f90
! RUN: bbc -emit-fir %t/cos.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/cos.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/cos.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/cos.f90
! RUN: bbc -emit-fir %t/cos.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/cos.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/cos.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/cos.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = cos(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.cos {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.cos {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @cosf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = cos(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.cos {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.cos {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @cos({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- cosh.f90
! RUN: bbc -emit-fir %t/cosh.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %t/cosh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/cosh.f90 -o - | FileCheck --check-prefixes=ALL %t/cosh.f90
! RUN: bbc -emit-fir %t/cosh.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %t/cosh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/cosh.f90 -o - | FileCheck --check-prefixes=ALL %t/cosh.f90
! RUN: bbc -emit-fir %t/cosh.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %t/cosh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/cosh.f90 -o - | FileCheck --check-prefixes=ALL %t/cosh.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = cosh(x)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @coshf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = cosh(x)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @cosh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- erf.f90
! RUN: bbc -emit-fir %t/erf.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/erf.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/erf.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/erf.f90
! RUN: bbc -emit-fir %t/erf.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/erf.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/erf.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/erf.f90
! RUN: bbc -emit-fir %t/erf.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/erf.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/erf.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/erf.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = erf(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.erf {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.erf {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @erff({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = erf(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.erf {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.erf {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @erf({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- exp.f90
! RUN: bbc -emit-fir %t/exp.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/exp.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/exp.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/exp.f90
! RUN: bbc -emit-fir %t/exp.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/exp.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/exp.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/exp.f90
! RUN: bbc -emit-fir %t/exp.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/exp.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/exp.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/exp.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = exp(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.exp {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.exp {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @expf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = exp(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.exp {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.exp {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @exp({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- floor.f90
! RUN: bbc -emit-fir %t/floor.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/floor.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/floor.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/floor.f90
! RUN: bbc -emit-fir %t/floor.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/floor.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/floor.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/floor.f90
! RUN: bbc -emit-fir %t/floor.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/floor.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/floor.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/floor.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = floor(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.floor {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.floor {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @floorf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = floor(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.floor {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.floor {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @floor({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- log.f90
! RUN: bbc -emit-fir %t/log.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/log.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/log.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/log.f90
! RUN: bbc -emit-fir %t/log.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/log.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/log.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/log.f90
! RUN: bbc -emit-fir %t/log.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/log.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/log.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/log.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = log(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.log {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.log {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @logf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = log(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.log {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.log {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @log({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- log10.f90
! RUN: bbc -emit-fir %t/log10.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/log10.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/log10.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/log10.f90
! RUN: bbc -emit-fir %t/log10.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/log10.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/log10.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/log10.f90
! RUN: bbc -emit-fir %t/log10.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/log10.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/log10.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/log10.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = log10(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.log10 {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.log10 {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @log10f({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = log10(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.log10 {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.log10 {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @log10({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- nint.f90
! RUN: bbc -emit-fir %t/nint.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %t/nint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/nint.f90 -o - | FileCheck --check-prefixes=ALL %t/nint.f90
! RUN: bbc -emit-fir %t/nint.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %t/nint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/nint.f90 -o - | FileCheck --check-prefixes=ALL %t/nint.f90
! RUN: bbc -emit-fir %t/nint.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %t/nint.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/nint.f90 -o - | FileCheck --check-prefixes=ALL %t/nint.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = nint(x, 4) + nint(x, 8)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.lround.i32.f32({{%[A-Za-z0-9._]+}}) : (f32) -> i32
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.lround.i64.f32({{%[A-Za-z0-9._]+}}) : (f32) -> i64

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = nint(x, 4) + nint(x, 8)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.lround.i32.f64({{%[A-Za-z0-9._]+}}) : (f64) -> i32
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.lround.i64.f64({{%[A-Za-z0-9._]+}}) : (f64) -> i64

//--- exponentiation.f90
! RUN: bbc -emit-fir %t/exponentiation.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/exponentiation.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/exponentiation.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/exponentiation.f90
! RUN: bbc -emit-fir %t/exponentiation.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/exponentiation.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/exponentiation.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/exponentiation.f90
! RUN: bbc -emit-fir %t/exponentiation.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/exponentiation.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/exponentiation.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/exponentiation.f90

function test_real4(x, y, s, i)
  real :: x, y, test_real4
  integer(2) :: s
  integer(4) :: i
  test_real4 = x ** s + x ** y + x ** i
end function

! ALL-LABEL: @_QPtest_real4
! ALL: [[STOI:%[A-Za-z0-9._]+]] = fir.convert {{%[A-Za-z0-9._]+}} : (i16) -> i32
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.powi.f32.i32({{%[A-Za-z0-9._]+}}, [[STOI]]) : (f32, i32) -> f32
! FAST: {{%[A-Za-z0-9._]+}} = math.powf {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.powf {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @powf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.powi.f32.i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, i32) -> f32

function test_real8(x, y, s, i)
  real(8) :: x, y, test_real8
  integer(2) :: s
  integer(4) :: i
  test_real8 = x ** s + x ** y + x ** i
end function

! ALL-LABEL: @_QPtest_real8
! ALL: [[STOI:%[A-Za-z0-9._]+]] = fir.convert {{%[A-Za-z0-9._]+}} : (i16) -> i32
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.powi.f64.i32({{%[A-Za-z0-9._]+}}, [[STOI]]) : (f64, i32) -> f64
! FAST: {{%[A-Za-z0-9._]+}} = math.powf {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.powf {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @pow({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @llvm.powi.f64.i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, i32) -> f64

//--- sign.f90
! RUN: bbc -emit-fir %t/sign.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/sign.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/sign.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/sign.f90
! RUN: bbc -emit-fir %t/sign.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/sign.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/sign.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/sign.f90
! RUN: bbc -emit-fir %t/sign.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/sign.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/sign.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/sign.f90

function test_real4(x, y)
  real :: x, y, test_real4
  test_real4 = sign(x, y)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @copysignf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

function test_real8(x, y)
  real(8) :: x, y, test_real8
  test_real8 = sign(x, y)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

function test_real10(x, y)
  real(10) :: x, y, test_real10
  test_real10 = sign(x, y)
end function

! ALL-LABEL: @_QPtest_real10
! FAST: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f80
! RELAXED: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f80
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @copysignl({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f80, f80) -> f80

function test_real16(x, y)
  real(16) :: x, y, test_real16
  test_real16 = sign(x, y)
end function

! ALL-LABEL: @_QPtest_real16
! FAST: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f128
! RELAXED: {{%[A-Za-z0-9._]+}} = math.copysign {{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}} : f128
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @llvm.copysign.f128({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f128, f128) -> f128

//--- sin.f90
! RUN: bbc -emit-fir %t/sin.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/sin.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/sin.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/sin.f90
! RUN: bbc -emit-fir %t/sin.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/sin.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/sin.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/sin.f90
! RUN: bbc -emit-fir %t/sin.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/sin.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/sin.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/sin.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = sin(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.sin {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.sin {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @sinf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = sin(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.sin {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.sin {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @sin({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- sinh.f90
! RUN: bbc -emit-fir %t/sinh.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %t/sinh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/sinh.f90 -o - | FileCheck --check-prefixes=ALL %t/sinh.f90
! RUN: bbc -emit-fir %t/sinh.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %t/sinh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/sinh.f90 -o - | FileCheck --check-prefixes=ALL %t/sinh.f90
! RUN: bbc -emit-fir %t/sinh.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %t/sinh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/sinh.f90 -o - | FileCheck --check-prefixes=ALL %t/sinh.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = sinh(x)
end function

! ALL-LABEL: @_QPtest_real4
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @sinhf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = sinh(x)
end function

! ALL-LABEL: @_QPtest_real8
! ALL: {{%[A-Za-z0-9._]+}} = fir.call @sinh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- tanh.f90
! RUN: bbc -emit-fir %t/tanh.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/tanh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/tanh.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/tanh.f90
! RUN: bbc -emit-fir %t/tanh.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/tanh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/tanh.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/tanh.f90
! RUN: bbc -emit-fir %t/tanh.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/tanh.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/tanh.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/tanh.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = tanh(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.tanh {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.tanh {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @tanhf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = tanh(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.tanh {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.tanh {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @tanh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

//--- tan.f90
! RUN: bbc -emit-fir %t/tan.f90 -o - --math-runtime=fast | FileCheck --check-prefixes=ALL,FAST %t/tan.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast %t/tan.f90 -o - | FileCheck --check-prefixes=ALL,FAST %t/tan.f90
! RUN: bbc -emit-fir %t/tan.f90 -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL,RELAXED %t/tan.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed %t/tan.f90 -o - | FileCheck --check-prefixes=ALL,RELAXED %t/tan.f90
! RUN: bbc -emit-fir %t/tan.f90 -o - --math-runtime=precise | FileCheck --check-prefixes=ALL,PRECISE %t/tan.f90
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise %t/tan.f90 -o - | FileCheck --check-prefixes=ALL,PRECISE %t/tan.f90

function test_real4(x)
  real :: x, test_real4
  test_real4 = tan(x)
end function

! ALL-LABEL: @_QPtest_real4
! FAST: {{%[A-Za-z0-9._]+}} = math.tan {{%[A-Za-z0-9._]+}} : f32
! RELAXED: {{%[A-Za-z0-9._]+}} = math.tan {{%[A-Za-z0-9._]+}} : f32
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @tanf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

function test_real8(x)
  real(8) :: x, test_real8
  test_real8 = tan(x)
end function

! ALL-LABEL: @_QPtest_real8
! FAST: {{%[A-Za-z0-9._]+}} = math.tan {{%[A-Za-z0-9._]+}} : f64
! RELAXED: {{%[A-Za-z0-9._]+}} = math.tan {{%[A-Za-z0-9._]+}} : f64
! PRECISE: {{%[A-Za-z0-9._]+}} = fir.call @tan({{%[A-Za-z0-9._]+}}) : (f64) -> f64

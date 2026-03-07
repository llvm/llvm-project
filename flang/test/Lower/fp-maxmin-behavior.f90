! Test lowering of real MIN/MAX with -ffp-maxmin-behavior (legacy, portable, extremum, extremenum).
! Legacy uses arith.cmpf + arith.select; extremum uses arith.maximumf/minimumf;
! extremenum uses arith.maxnumf/minnumf; portable with -fno-signed-zeros -menable-no-nans uses maxnumf/minnumf.

! bbc: legacy, extremum, extremenum
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s -check-prefix=LEGACY
! RUN: bbc -emit-hlfir -ffp-maxmin-behavior=legacy -o - %s 2>&1 | FileCheck %s -check-prefix=LEGACY
! RUN: bbc -emit-hlfir -ffp-maxmin-behavior=extremum -o - %s 2>&1 | FileCheck %s -check-prefix=EXTREMUM
! RUN: bbc -emit-hlfir -ffp-maxmin-behavior=extremenum -o - %s 2>&1 | FileCheck %s -check-prefix=EXTREMENUM

! flang -fc1: legacy, extremum, extremenum
! RUN: %flang_fc1 -emit-hlfir -o - %s 2>&1 | FileCheck %s -check-prefix=LEGACY
! RUN: %flang_fc1 -emit-hlfir -ffp-maxmin-behavior=legacy -o - %s 2>&1 | FileCheck %s -check-prefix=LEGACY
! RUN: %flang_fc1 -emit-hlfir -ffp-maxmin-behavior=extremum -o - %s 2>&1 | FileCheck %s -check-prefix=EXTREMUM
! RUN: %flang_fc1 -emit-hlfir -ffp-maxmin-behavior=extremenum -o - %s 2>&1 | FileCheck %s -check-prefix=EXTREMENUM

! portable with -fno-signed-zeros -menable-no-nans => maxnumf/minnumf (flang -fc1 only; bbc does not expose these flags)
! RUN: %flang_fc1 -emit-hlfir -ffp-maxmin-behavior=portable -fno-signed-zeros -menable-no-nans -o - %s 2>&1 | FileCheck %s -check-prefix=PORTABLE-NANNSZ

subroutine real_max(a, b, r)
  real :: a, b, r
  r = max(a, b)
end subroutine
! LEGACY-LABEL: func.func @_QPreal_max(
! LEGACY: arith.cmpf ogt,
! LEGACY: arith.select

! EXTREMUM-LABEL: func.func @_QPreal_max(
! EXTREMUM: arith.maximumf

! EXTREMENUM-LABEL: func.func @_QPreal_max(
! EXTREMENUM: arith.maxnumf

! PORTABLE-NANNSZ-LABEL: func.func @_QPreal_max(
! PORTABLE-NANNSZ: arith.maxnumf

subroutine real_min(a, b, r)
  real :: a, b, r
  r = min(a, b)
end subroutine
! LEGACY-LABEL: func.func @_QPreal_min(
! LEGACY: arith.cmpf olt,
! LEGACY: arith.select

! EXTREMUM-LABEL: func.func @_QPreal_min(
! EXTREMUM: arith.minimumf

! EXTREMENUM-LABEL: func.func @_QPreal_min(
! EXTREMENUM: arith.minnumf

! PORTABLE-NANNSZ-LABEL: func.func @_QPreal_min(
! PORTABLE-NANNSZ: arith.minnumf

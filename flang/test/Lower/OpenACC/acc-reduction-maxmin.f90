! This test checks lowering of OpenACC reduction clause.

! RUN: %flang_fc1 -fopenacc -emit-hlfir -o - %s 2>&1 | FileCheck %s -check-prefix=LEGACY
! RUN: %flang_fc1 -fopenacc -emit-hlfir -ffp-maxmin-behavior=legacy -o - %s 2>&1 | FileCheck %s -check-prefix=LEGACY
! RUN: %flang_fc1 -fopenacc -emit-hlfir -ffp-maxmin-behavior=extremum -o - %s 2>&1 | FileCheck %s -check-prefix=EXTREMUM
! RUN: %flang_fc1 -fopenacc -emit-hlfir -ffp-maxmin-behavior=extremenum -o - %s 2>&1 | FileCheck %s -check-prefix=EXTREMENUM

! TODO: we should get rid of the legacy mode to make the generation of
! arith.max/minnumf straightforward for portable mode + nsz + nnan:
! RUN: %flang_fc1 -fopenacc -emit-hlfir -ffp-maxmin-behavior=portable -fno-signed-zeros -menable-no-nans -o - %s 2>&1 | FileCheck %s -check-prefix=PORTABLE-NANNSZ

subroutine acc_scalar_reduction_max(a)
  real :: a
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine acc_scalar_reduction_max

subroutine acc_array_reduction_max(a)
  real :: a(10)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine acc_array_reduction_max

subroutine acc_scalar_reduction_min(a)
  real :: a
  !$acc parallel reduction(min:a)
  !$acc end parallel
end subroutine acc_scalar_reduction_min

subroutine acc_array_reduction_min(a)
  real :: a(10)
  !$acc parallel reduction(min:a)
  !$acc end parallel
end subroutine acc_array_reduction_min

! LEGACY-LABEL:   acc.reduction.recipe @reduction_min_ref_10xf32 : !fir.ref<!fir.array<10xf32>> reduction_operator <min> init {
! LEGACY:         } combiner {
! LEGACY:           fir.do_loop
! LEGACY:             %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[LOAD_0:.*]] fastmath<contract> : f32
! LEGACY:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32

! LEGACY-LABEL:   acc.reduction.recipe @reduction_min_ref_f32 : !fir.ref<f32> reduction_operator <min> init {
! LEGACY:         } combiner {
! LEGACY:           %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[LOAD_0:.*]] fastmath<contract> : f32
! LEGACY:           %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32

! LEGACY-LABEL:   acc.reduction.recipe @reduction_max_ref_10xf32 : !fir.ref<!fir.array<10xf32>> reduction_operator <max> init {
! LEGACY:         } combiner {
! LEGACY:           fir.do_loop
! LEGACY:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[LOAD_0:.*]] fastmath<contract> : f32
! LEGACY:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32

! LEGACY-LABEL:   acc.reduction.recipe @reduction_max_ref_f32 : !fir.ref<f32> reduction_operator <max> init {
! LEGACY:         } combiner {
! LEGACY:           %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[LOAD_0:.*]] fastmath<contract> : f32
! LEGACY:           %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32

! EXTREMUM-LABEL:   acc.reduction.recipe @reduction_minimumf_ref_10xf32 : !fir.ref<!fir.array<10xf32>> reduction_operator <minimumf> init {
! EXTREMUM:         } combiner {
! EXTREMUM:           fir.do_loop
! EXTREMUM:             %[[MINIMUMF_0:.*]] = arith.minimumf %{{.*}}, %{{.*}} fastmath<contract> : f32

! EXTREMUM-LABEL:   acc.reduction.recipe @reduction_minimumf_ref_f32 : !fir.ref<f32> reduction_operator <minimumf> init {
! EXTREMUM:           %[[CST:.*]] = arith.constant 3.40282347E+38 : f32
! EXTREMUM:         } combiner {
! EXTREMUM:           %[[MINIMUMF_0:.*]] = arith.minimumf %{{.*}}, %{{.*}} fastmath<contract> : f32

! EXTREMUM-LABEL:   acc.reduction.recipe @reduction_maximumf_ref_10xf32 : !fir.ref<!fir.array<10xf32>> reduction_operator <maximumf> init {
! EXTREMUM:         } combiner {
! EXTREMUM:           fir.do_loop
! EXTREMUM:             %[[MAXIMUMF_0:.*]] = arith.maximumf %{{.*}}, %{{.*}} fastmath<contract> : f32

! EXTREMUM-LABEL:   acc.reduction.recipe @reduction_maximumf_ref_f32 : !fir.ref<f32> reduction_operator <maximumf> init {
! EXTREMUM-LABEL:   } combiner {
! EXTREMUM:           %[[MAXIMUMF_0:.*]] = arith.maximumf %{{.*}}, %{{.*}} fastmath<contract> : f32

! EXTREMENUM-LABEL:   acc.reduction.recipe @reduction_minnumf_ref_10xf32 : !fir.ref<!fir.array<10xf32>> reduction_operator <minnumf> init {
! EXTREMENUM:         } combiner {
! EXTREMENUM:           fir.do_loop
! EXTREMENUM:             %[[MINNUMF_0:.*]] = arith.minnumf %{{.*}}, %{{.*}} fastmath<contract> : f32

! EXTREMENUM-LABEL:   acc.reduction.recipe @reduction_minnumf_ref_f32 : !fir.ref<f32> reduction_operator <minnumf> init {
! EXTREMENUM:           %[[CST:.*]] = arith.constant 3.40282347E+38 : f32
! EXTREMENUM:         } combiner {
! EXTREMENUM:           %[[MINNUMF_0:.*]] = arith.minnumf %{{.*}}, %{{.*}} fastmath<contract> : f32

! EXTREMENUM-LABEL:   acc.reduction.recipe @reduction_maxnumf_ref_10xf32 : !fir.ref<!fir.array<10xf32>> reduction_operator <maxnumf> init {
! EXTREMENUM:         } combiner {
! EXTREMENUM:           fir.do_loop
! EXTREMENUM:             %[[MAXNUMF_0:.*]] = arith.maxnumf %{{.*}}, %{{.*}} fastmath<contract> : f32

! EXTREMENUM-LABEL:   acc.reduction.recipe @reduction_maxnumf_ref_f32 : !fir.ref<f32> reduction_operator <maxnumf> init {
! EXTREMENUM-LABEL:   } combiner {
! EXTREMENUM:           %[[MAXNUMF_0:.*]] = arith.maxnumf %{{.*}}, %{{.*}} fastmath<contract> : f32

! PORTABLE-NANNSZ-LABEL:   acc.reduction.recipe @reduction_min_ref_10xf32 : !fir.ref<!fir.array<10xf32>> reduction_operator <min> init {
! PORTABLE-NANNSZ:         } combiner {
! PORTABLE-NANNSZ:           fir.do_loop
! PORTABLE-NANNSZ:             %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[LOAD_0:.*]] fastmath<nnan,nsz,contract> : f32
! PORTABLE-NANNSZ:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32

! PORTABLE-NANNSZ-LABEL:   acc.reduction.recipe @reduction_min_ref_f32 : !fir.ref<f32> reduction_operator <min> init {
! PORTABLE-NANNSZ:         } combiner {
! PORTABLE-NANNSZ:           %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1:.*]], %[[LOAD_0:.*]] fastmath<nnan,nsz,contract> : f32
! PORTABLE-NANNSZ:           %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32

! PORTABLE-NANNSZ-LABEL:   acc.reduction.recipe @reduction_max_ref_10xf32 : !fir.ref<!fir.array<10xf32>> reduction_operator <max> init {
! PORTABLE-NANNSZ:         } combiner {
! PORTABLE-NANNSZ:           fir.do_loop
! PORTABLE-NANNSZ:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[LOAD_0:.*]] fastmath<nnan,nsz,contract> : f32
! PORTABLE-NANNSZ:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32

! PORTABLE-NANNSZ-LABEL:   acc.reduction.recipe @reduction_max_ref_f32 : !fir.ref<f32> reduction_operator <max> init {
! PORTABLE-NANNSZ:         } combiner {
! PORTABLE-NANNSZ:           %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1:.*]], %[[LOAD_0:.*]] fastmath<nnan,nsz,contract> : f32
! PORTABLE-NANNSZ:           %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32

! This test checks lowering of OpenMP Threadprivate Directive.
! Test for common block, defined in one module, used in a subroutine of
! another module and privatized in a nested subroutine.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: fir.global common @cmn_(dense<0> : vector<4xi8>) : !fir.array<4xi8>
module m0
  common /cmn/ k1
  !$omp threadprivate(/cmn/)
end

module  m1
contains
  subroutine ss1
    use m0
  contains
!CHECK-LABEL: func @_QMm1Fss1Pss2
!CHECK: %[[CMN:.*]] = fir.address_of(@cmn_) : !fir.ref<!fir.array<4xi8>>
!CHECK: omp.parallel
!CHECK: %{{.*}} = omp.threadprivate %[[CMN]] : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
    subroutine ss2
      !$omp parallel copyin (k1)
      !$omp end parallel
    end subroutine ss2
  end subroutine ss1
end

end

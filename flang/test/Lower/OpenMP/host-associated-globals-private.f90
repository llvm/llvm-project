! REQUIRES: module-independent
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! Canonical SAVE globals remain accessible through sibling calls. Each OpenMP
! copy has separate storage and must not inherit internal_assoc from its source.
! CHECK-LABEL: func.func private @_QFhPworker()
! CHECK-DAG: hlfir.declare {{.*}}uniq_name("_QFhEx") fortran_attrs<internal_assoc>
! CHECK-DAG: hlfir.declare {{.*}}uniq_name("_QFhEy") fortran_attrs<internal_assoc>
! CHECK-DAG: hlfir.declare {{.*}}uniq_name("_QFhEz") fortran_attrs<internal_assoc>
! CHECK: omp.parallel
! CHECK-DAG: hlfir.declare %arg0 uniq_name("_QFhEx") :
! CHECK-DAG: hlfir.declare %arg1 uniq_name("_QFhEy") :
! CHECK-DAG: hlfir.declare %arg2 uniq_name("_QFhEz") :
! CHECK-LABEL: func.func private @_QFhPbump()
! CHECK-DAG: hlfir.declare {{.*}}uniq_name("_QFhEx") fortran_attrs<internal_assoc>
! CHECK-DAG: hlfir.declare {{.*}}uniq_name("_QFhEy") fortran_attrs<internal_assoc>
! CHECK-DAG: hlfir.declare {{.*}}uniq_name("_QFhEz") fortran_attrs<internal_assoc>
subroutine h
  integer, save :: x,y,z
  x=7
  y=8
  z=9
  call worker()
contains
  subroutine worker()
!$omp parallel num_threads(1) firstprivate(x) private(y) reduction(+:z)
    call bump()
    y=x
    z=z+y
    print *,x,y,z
!$omp end parallel
  end subroutine
  subroutine bump()
    x=x+1
    y=y+1
    z=z+1
  end subroutine
end subroutine

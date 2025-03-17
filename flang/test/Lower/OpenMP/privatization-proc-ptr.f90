! Test privatization of procedure pointers.

!RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

program proc_ptr_test
  implicit none

contains

!CHECK: omp.private {type = private} @_QFFtest_namesEpf2_private_boxproc_z32_args_ref_3x4xf32_ref_z32 : !fir.boxproc<(!fir.ref<!fir.array<3x4xf32>>, !fir.ref<complex<f32>>) -> complex<f32>>
!CHECK: omp.private {type = private} @_QFFtest_namesEpf1_private_boxproc_f32_args_ref_f32 : !fir.boxproc<(!fir.ref<f32>) -> f32>
!CHECK: omp.private {type = private} @_QFFtest_namesEpf0_private_boxproc_i32_args : !fir.boxproc<() -> i32>
!CHECK: omp.private {type = private} @_QFFtest_namesEps2_private_boxproc__args_ref_i32_boxchar_c8xU : !fir.boxproc<(!fir.ref<i32>, !fir.boxchar<1>) -> ()>
!CHECK: omp.private {type = private} @_QFFtest_namesEps1_private_boxproc__args_ref_i32 : !fir.boxproc<(!fir.ref<i32>) -> ()>
!CHECK: omp.private {type = private} @_QFFtest_namesEps0_private_boxproc__args : !fir.boxproc<() -> ()>

!CHECK: omp.private {type = private} @_QFFtest_lastprivateEps_private_boxproc__args_ref_i32 : !fir.boxproc<(!fir.ref<i32>) -> ()>
!CHECK: omp.private {type = private} @_QFFtest_lastprivateEpf_private_boxproc_i32_args_ref_i32 : !fir.boxproc<(!fir.ref<i32>) -> i32>

!CHECK: omp.private {type = firstprivate} @_QFFtest_firstprivateEps_firstprivate_boxproc__args_ref_i32 : !fir.boxproc<(!fir.ref<i32>) -> ()> copy {
!CHECK:  ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> ()>>, %[[ARG1:.*]]: !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> ()>>):
!CHECK:    %[[TEMP:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> ()>>
!CHECK:    fir.store %[[TEMP]] to %[[ARG1]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> ()>>
!CHECK:    omp.yield(%[[ARG1]] : !fir.ref<!fir.boxproc<(!fir.ref<i32>) -> ()>>)
!CHECK: }

!CHECK: omp.private {type = firstprivate} @_QFFtest_firstprivateEpf_firstprivate_boxproc_i32_args_ref_i32 : !fir.boxproc<(!fir.ref<i32>) -> i32>
!CHECK: omp.private {type = private} @_QFFtest_privateEps_private_boxproc__args_ref_i32 : !fir.boxproc<(!fir.ref<i32>) -> ()>
!CHECK: omp.private {type = private} @_QFFtest_privateEpf_private_boxproc_i32_args_ref_i32 : !fir.boxproc<(!fir.ref<i32>) -> i32>

!CHECK-LABEL: func private @_QFPtest_private
!CHECK:       %[[PF:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_privateEpf"}
!CHECK:       %[[PS:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_privateEps"}
!CHECK:       omp.parallel
!CHECK:         %[[PRIV_PF:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_privateEpf"}
!CHECK:         %[[PRIV_PS:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_privateEps"}
!CHECK:         %[[PF_VAL:.*]] = fir.load %[[PRIV_PF]]#0
!CHECK:         %[[PF_BOX:.*]] = fir.box_addr %[[PF_VAL]]
!CHECK:         fir.call %[[PF_BOX]]({{.*}})
!CHECK:         %[[PS_VAL:.*]] = fir.load %[[PRIV_PS]]#0
!CHECK:         %[[PS_BOX:.*]] = fir.box_addr %[[PS_VAL]]
!CHECK:         fir.call %[[PS_BOX]]({{.*}})
subroutine test_private
  procedure(f), pointer :: pf
  procedure(sub), pointer :: ps
  integer :: res

  !$omp parallel private(pf, ps)
    pf => f
    ps => sub
    res = pf(123)
    call ps(456)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func private @_QFPtest_firstprivate
!CHECK:       %[[PF:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_firstprivateEpf"}
!CHECK:       %[[PS:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_firstprivateEps"}
!CHECK:       omp.parallel
!CHECK:         %[[PRIV_PF:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_firstprivateEpf"}
!CHECK:         %[[PRIV_PS:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_firstprivateEps"}
subroutine test_firstprivate
  procedure(f), pointer :: pf
  procedure(sub), pointer :: ps

  !$omp parallel firstprivate(pf, ps)
  !$omp end parallel
end subroutine

!CHECK-LABEL: func private @_QFPtest_lastprivate
!CHECK:       %[[PF:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_lastprivateEpf"}
!CHECK:       %[[PS:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_lastprivateEps"}
!CHECK:       omp.parallel
!CHECK:         %[[PRIV_PF:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_lastprivateEpf"}
!CHECK:         %[[PRIV_PS:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_lastprivateEps"}
!CHECK:         %[[PF_VAL:.*]] = fir.load %[[PRIV_PF]]#0
!CHECK:         fir.store %[[PF_VAL]] to %[[PF]]#0
!CHECK:         %[[PS_VAL:.*]] = fir.load %[[PRIV_PS]]#0
!CHECK:         fir.store %[[PS_VAL]] to %[[PS]]#0
subroutine test_lastprivate
  procedure(f), pointer :: pf
  procedure(sub), pointer :: ps
  integer :: i

  !$omp parallel do lastprivate(pf, ps)
  do i = 1, 5
  end do
  !$omp end parallel do
end subroutine

!CHECK-LABEL: func private @_QFPtest_sections
!CHECK:       %[[PF:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_sectionsEpf"}
!CHECK:       %[[PS:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_sectionsEps"}
!CHECK:       %[[PRIV_PF:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_sectionsEpf"}
!CHECK:       %[[PF_VAL:.*]] = fir.load %[[PF]]#0
!CHECK:       fir.store %[[PF_VAL]] to %[[PRIV_PF]]#0
!CHECK:       %[[PRIV_PS:.*]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QFFtest_sectionsEps"}
!CHECK:       %[[PS_VAL:.*]] = fir.load %[[PS]]#0
!CHECK:       fir.store %[[PS_VAL]] to %[[PRIV_PS]]#0
!CHECK:       omp.sections
!CHECK:         %[[PF_VAL:.*]] = fir.load %[[PRIV_PF]]#0
!CHECK:         fir.store %[[PF_VAL]] to %[[PF]]#0
!CHECK:         %[[PS_VAL:.*]] = fir.load %[[PRIV_PS]]#0
!CHECK:         fir.store %[[PS_VAL]] to %[[PS]]#0
subroutine test_sections
  procedure(f), pointer :: pf
  procedure(sub), pointer :: ps

  !$omp sections firstprivate(pf, ps) lastprivate(pf, ps)
  !$omp end sections
end subroutine

integer function f(arg)
  integer :: arg
  f = arg
end function

subroutine sub(arg)
  integer :: arg
end subroutine

subroutine test_names
  procedure(s0), pointer :: ps0
  procedure(s1), pointer :: ps1
  procedure(s2), pointer :: ps2

  procedure(f0), pointer :: pf0
  procedure(f1), pointer :: pf1
  procedure(f2), pointer :: pf2

  !$omp parallel private(ps0, ps1, ps2, pf0, pf1, pf2)
  !$omp end parallel
end subroutine

subroutine s0
end subroutine

subroutine s1(i)
  integer :: i
end subroutine

subroutine s2(i, j)
  integer :: i
  character(*) :: j
end subroutine

integer function f0
  f0 = 0
end function

real function f1(r)
  real :: r

  f1 = 0.0
end function

function f2(a, c)
  real :: a(3, 4)
  complex :: f2, c

  f2 = (0.0, 0.0)
end function

end program

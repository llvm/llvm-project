! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
  integer :: n
end module

program p
  integer :: i

  !$omp parallel do
  do i=1,2
    block
      use m
      do n=1,2
      end do
    end block
  end do
end program p

! Verify the privatizer recipe for the module variable is created.
! CHECK: omp.private {type = private} @[[N_PRIV:.*QMmEn_private.*]] : i32
! CHECK: omp.private {type = private} @[[I_PRIV:.*Ei_private.*]] : i32

! Verify the module global exists.
! CHECK: fir.global @_QMmEn : i32

! CHECK-LABEL: func.func @_QQmain()
! CHECK:         %[[I_ALLOC:.*]] = fir.alloca i32 {bindc_name = "i"
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ALLOC]] {uniq_name = "_QFEi"}
! CHECK:         omp.parallel {

! Verify the module variable is instantiated inside the parallel region.
! CHECK:           %[[N_ADDR:.*]] = fir.address_of(@_QMmEn) : !fir.ref<i32>
! CHECK:           %[[N_DECL:.*]]:2 = hlfir.declare %[[N_ADDR]] {uniq_name = "_QMmEn"}

! Verify the wsloop privatizes both i and n.
! CHECK:           omp.wsloop private(@[[I_PRIV]] %[[I_DECL]]#0 -> %{{.*}}, @[[N_PRIV]] %[[N_DECL]]#0 -> %{{.*}} : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:             omp.loop_nest (%{{.*}}) : i32 =
! CHECK:               fir.do_loop
! CHECK:               omp.yield
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }

! RUN: %flang_fc1 -emit-hlfir -mmlir --enable-delayed-privatization-staging=true -o - %s | FileCheck %s

subroutine loop_in_nested_block
  implicit none
  integer :: i, j

  do concurrent (i=1:10) local(j)
    block
      do j=1,20
      end do
    end block
  end do
end subroutine

! CHECK-LABEL: func.func @_QPloop_in_nested_block() {
! CHECK:         %[[OUTER_J_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Ej"}
! CHECK:         fir.do_concurrent {
! CHECK:           fir.do_concurrent.loop {{.*}} local(@{{.*}} %[[OUTER_J_DECL]]#0 -> %[[LOCAL_J_ARG:.*]] : !fir.ref<i32>) {
! CHECK:             %[[LOCAL_J_DECL:.*]]:2 = hlfir.declare %[[LOCAL_J_ARG]]
! CHECK:             fir.do_loop {{.*}} iter_args(%[[NESTED_LOOP_ARG:.*]] = {{.*}}) {
! CHECK:               fir.store %[[NESTED_LOOP_ARG]] to %[[LOCAL_J_DECL]]#0
! CHECK:             }
! CHECK:           }
! CHECK:         }
! CHECK:       }


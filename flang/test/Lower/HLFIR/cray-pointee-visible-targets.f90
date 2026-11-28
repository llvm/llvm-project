! Cray pointee references are re-associated with fir.box_set_addr, which is a
! plain store LLVM can see through. Accesses to the entities a Cray pointer
! may visibly point to must then be ordered with the pointee accesses by
! their TARGET attribute. Check that the targets of these idioms are marked.
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

subroutine putaddr(p, q)
  integer(8) :: p, q
  p = q
end subroutine

integer function equivalence_partner()
  integer :: a(10), b(10)
  equivalence (a, b)
  integer :: x
  pointer (px, x)
  integer :: k
  a = 1
  px = loc(a(5))
  do k = 1, 10
    b(k) = b(k) + x
  end do
  equivalence_partner = a(9)
end function
! CHECK-LABEL: func.func @_QPequivalence_partner()
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFequivalence_partnerEa") fortran_attrs<target
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFequivalence_partnerEb") fortran_attrs<target
! CHECK-DAG:     %[[X:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFequivalence_partnerEx") fortran_attrs<pointer, cray_pointee>
! CHECK:         fir.do_loop
! CHECK:           fir.box_set_addr %{{.*}} to %[[X]]#0

integer function loc_plus_offset()
  integer :: a(10)
  integer :: x
  pointer (px, x)
  integer :: k
  a = 1
  px = loc(a) + 16
  do k = 1, 10
    a(k) = a(k) + x
  end do
  loc_plus_offset = a(9)
end function
! CHECK-LABEL: func.func @_QPloc_plus_offset()
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFloc_plus_offsetEa") fortran_attrs<target
! CHECK-DAG:     %[[X:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFloc_plus_offsetEx") fortran_attrs<pointer, cray_pointee>
! CHECK:         fir.do_loop
! CHECK:           fir.box_set_addr %{{.*}} to %[[X]]#0

integer function loc_actual_argument()
  integer :: a(10), x, k
  pointer (px, x)
  a = 1
  call putaddr(px, loc(a(5)))
  do k = 1, 10
    a(k) = a(k) + x
  end do
  loc_actual_argument = a(9)
end function
! CHECK-LABEL: func.func @_QPloc_actual_argument()
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFloc_actual_argumentEa") fortran_attrs<target
! CHECK-DAG:     %[[X:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFloc_actual_argumentEx") fortran_attrs<pointer, cray_pointee>
! CHECK:         fir.do_loop
! CHECK:           fir.box_set_addr %{{.*}} to %[[X]]#0

integer function common_block_next_member()
  integer :: a(5), b(5)
  common /blk/ a, b
  integer :: x
  pointer (px, x)
  integer :: k
  a = 1
  b = 1
  px = loc(a(5)) + 4
  do k = 1, 5
    b(k) = b(k) + x
  end do
  common_block_next_member = b(5)
end function
! CHECK-LABEL: func.func @_QPcommon_block_next_member()
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFcommon_block_next_memberEa") fortran_attrs<target
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFcommon_block_next_memberEb") fortran_attrs<target
! CHECK-DAG:     %[[X:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFcommon_block_next_memberEx") fortran_attrs<pointer, cray_pointee>
! CHECK:         fir.do_loop
! CHECK:           fir.box_set_addr %{{.*}} to %[[X]]#0

integer function loc_through_integer()
  integer :: a(10), x, k
  integer(8) :: iaddr
  pointer (px, x)
  a = 1
  iaddr = loc(a(5))
  px = iaddr
  do k = 1, 10
    a(k) = a(k) + x
  end do
  loc_through_integer = a(9)
end function
! CHECK-LABEL: func.func @_QPloc_through_integer()
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFloc_through_integerEa") fortran_attrs<target
! CHECK-DAG:     %[[X:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFloc_through_integerEx") fortran_attrs<pointer, cray_pointee>
! CHECK:         fir.do_loop
! CHECK:           fir.box_set_addr %{{.*}} to %[[X]]#0

integer function loc_in_internal_procedure()
  integer :: a(10), x, k
  pointer (px, x)
  a = 1
  call associate()
  do k = 1, 10
    a(k) = a(k) + x
  end do
  loc_in_internal_procedure = a(9)
contains
  subroutine associate()
    px = loc(a(5))
  end subroutine
end function
! CHECK-LABEL: func.func @_QPloc_in_internal_procedure()
! CHECK-DAG:     hlfir.declare {{.*}} uniq_name("_QFloc_in_internal_procedureEa") fortran_attrs<target
! CHECK-DAG:     %[[X:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFloc_in_internal_procedureEx") fortran_attrs<pointer, cray_pointee>
! CHECK:         fir.do_loop
! CHECK:           fir.box_set_addr %{{.*}} to %[[X]]#0

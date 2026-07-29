! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module impure_mod
contains
  pure integer function pure_bound(k)
    integer, intent(in) :: k
    pure_bound = k
  end function
  integer function impure_bound(k)
    integer, intent(in) :: k
    impure_bound = k
  end function
end module

! CHECK-LABEL: func @_QPwrite_whole(
subroutine write_whole(a, n)
  integer :: n
  real :: a(n)
  ! CHECK: %[[SEC:.*]] = hlfir.designate %{{.*}} (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[BOX:.*]] = fir.convert %[[SEC]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX]])
  ! CHECK-NOT: fir.call @_FortranAioOutputReal
  ! CHECK-NOT: fir.do_loop
  write(10) (a(i), i=1,n)
end subroutine

! CHECK-LABEL: func @_QPread_whole(
subroutine read_whole(a, n)
  integer :: n
  real :: a(n)
  ! CHECK: %[[SEC:.*]] = hlfir.designate %{{.*}} (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[BOX:.*]] = fir.convert %[[SEC]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioInputDescriptor(%{{.*}}, %[[BOX]])
  ! CHECK-NOT: fir.call @_FortranAioInputReal
  ! CHECK-NOT: fir.do_loop
  read(10) (a(i), i=1,n)
end subroutine

! CHECK-LABEL: func @_QPwrite_finalval(
subroutine write_finalval(a, n, k)
  integer :: n, k
  real :: a(n)
  ! CHECK: %[[SEC:.*]] = hlfir.designate %{{.*}} (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[SEC]], %{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  ! CHECK: %[[TS:.*]] = arith.muli %[[DIMS]]#1, %{{.*}} : index
  ! CHECK: %[[FV:.*]] = arith.addi %{{.*}}, %[[TS]] : index
  ! CHECK: %[[FVC:.*]] = fir.convert %[[FV]] : (index) -> i32
  ! CHECK: fir.store %[[FVC]] to %{{.*}} : !fir.ref<i32>
  write(10) (a(i), i=1,n)
  k = i
end subroutine

! CHECK-LABEL: func @_QPwrite_fixed(
subroutine write_fixed(b, n)
  integer :: n
  real :: b(5,n)
  ! CHECK: hlfir.designate %{{.*}} (%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<5x?xf32>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK-NOT: fir.call @_FortranAioOutputReal
  write(10) (b(3,i), i=1,n)
end subroutine

! CHECK-LABEL: func @_QPwrite_step_valid(
subroutine write_step_valid(a, n)
  integer :: n
  real :: a(n)
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK-NOT: fir.call @_FortranAioOutputReal
  ! CHECK-NOT: fir.do_loop
  write(10) (a(i), i=1,n,2)
end subroutine

! CHECK-LABEL: func @_QPwrite_pure_bound(
subroutine write_pure_bound(a, n)
  use impure_mod
  integer :: n
  real :: a(n)
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK-NOT: fir.call @_FortranAioOutputReal
  ! CHECK-NOT: fir.do_loop
  write(10) (a(i), i=1,pure_bound(n))
end subroutine

! CHECK-LABEL: func @_QPwrite_alias_subscript(
subroutine write_alias_subscript(b, n)
  integer :: n
  integer :: b(10, n)
  ! CHECK: hlfir.designate %{{.*}} (%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<10x?xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK-NOT: fir.do_loop
  write(10) (b(b(1,1), i), i=1,n)
end subroutine

! ===========================================================================
! The cases below are NOT collapsed and fall back to a per-element loop.
! ===========================================================================

! Formatted transfer needs per-element edit  descriptors.
! CHECK-LABEL: func @_QPwrite_formatted(
subroutine write_formatted(a, n)
  integer :: n
  real :: a(n)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputReal
  write(10,*) (a(i), i=1,n)
end subroutine

! Condition specifier (IOSTAT=) needs per-iteration error handling.
! CHECK-LABEL: func @_QPwrite_iostat(
subroutine write_iostat(a, n, ios)
  integer :: n, ios
  real :: a(n)
  ! CHECK: fir.iterate_while
  ! A per-element scalar embox inside the loop confirms this was not collapsed;
  ! a collapsed transfer would designate a whole array section instead.
  ! CHECK: fir.embox %{{.*}} : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10, iostat=ios) (a(i), i=1,n)
end subroutine

! Non-identity subscript a(2*i) is not a simple section.
! CHECK-LABEL: func @_QPwrite_nonidentity(
subroutine write_nonidentity(a, n)
  integer :: n
  real :: a(2*n)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(2*i), i=1,n)
end subroutine

! Step references the loop variable: no static triplet stride.
! CHECK-LABEL: func @_QPwrite_step_invalid(
subroutine write_step_invalid(a, n)
  integer :: n
  real :: a(n)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(i), i=1,n,i)
end subroutine

! Impure bound may not be evaluated a different number of times.
! CHECK-LABEL: func @_QPwrite_impure_bound(
subroutine write_impure_bound(a, n)
  use impure_mod
  integer :: n
  real :: a(n)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(i), i=1,impure_bound(n))
end subroutine

! Loop variable used in more than one subscript, b(i,i).
! CHECK-LABEL: func @_QPwrite_loopvar_twice(
subroutine write_loopvar_twice(b, n)
  integer :: n
  real :: b(n,n)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (b(i,i), i=1,n)
end subroutine

! Loop variable not used in any subscript, b(3,4).
! CHECK-LABEL: func @_QPwrite_no_loopvar(
subroutine write_no_loopvar(b, n)
  integer :: n
  real :: b(5,5)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (b(3,4), i=1,n)
end subroutine

! Item is an array section (whole-dimension ':' subscript), not an element.
! CHECK-LABEL: func @_QPwrite_section_whole_dim(
subroutine write_section_whole_dim(a, n)
  integer :: n
  real :: a(3,n)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(:, i), i=1,n)
end subroutine

! Item is an array section (explicit triplet subscript).
! CHECK-LABEL: func @_QPwrite_section_triplet(
subroutine write_section_triplet(a, n)
  integer :: n
  real :: a(3,n)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(1:3, i), i=1,n)
end subroutine

! Triplet in another dimension still makes the item a section.
! CHECK-LABEL: func @_QPwrite_section_trailing(
subroutine write_section_trailing(a, n)
  integer :: n
  real :: a(n,5)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(i, 2:4), i=1,n)
end subroutine

! Volatile bound could be read a different number of times.
! CHECK-LABEL: func @_QPwrite_volatile_bound(
subroutine write_volatile_bound(a, lo)
  integer, volatile :: lo
  real :: a(100)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(i), i=lo,50)
end subroutine

! Volatile retained subscript could be read a different number of times.
! CHECK-LABEL: func @_QPwrite_volatile_subscript(
subroutine write_volatile_subscript(b, n, j)
  integer :: n
  integer, volatile :: j
  real :: b(10,100)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (b(j,i), i=1,n)
end subroutine

! Input with a retained subscript that references the array being read into:
! a real per-element loop re-evaluates b(1,1) each iteration and so observes
! values stored by earlier iterations, while a collapsed section would evaluate
! it once. This must NOT be collapsed.
! CHECK-LABEL: func @_QPread_alias_subscript(
subroutine read_alias_subscript(b, n)
  integer :: n
  integer :: b(10, n)
  ! CHECK: fir.do_loop
  ! CHECK: fir.embox %{{.*}} : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: fir.call @_FortranAioInputDescriptor
  read(10) (b(b(1,1), i), i=1,n)
end subroutine

! Loop variable is EQUIVALENCEd to an element of the array being written.
! CHECK-LABEL: func @_QPwrite_equiv_loopvar(
subroutine write_equiv_loopvar()
  integer :: a(8), i
  equivalence (i, a(4))
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(i), i=1,8)
end subroutine

! Loop variable is a POINTER associated to an element of the array.
! CHECK-LABEL: func @_QPwrite_pointer_loopvar(
subroutine write_pointer_loopvar()
  integer, target :: a(8)
  integer, pointer :: i
  i => a(4)
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(i), i=1,8)
end subroutine

! Loop variable is an ASSOCIATE construct entity aliasing an element.
! CHECK-LABEL: func @_QPwrite_associate_loopvar(
subroutine write_associate_loopvar(a)
  integer :: a(8)
  associate (i => a(4))
    ! CHECK: fir.do_loop
    ! CHECK: fir.call @_FortranAioOutputDescriptor
    write(10) (a(i), i=1,8)
  end associate
end subroutine

! Loop variable is a dummy argument that may be argument associated with the
! array (e.g. call sub(b, b(4))).
! CHECK-LABEL: func @_QPwrite_dummy_loopvar(
subroutine write_dummy_loopvar(a, i)
  integer :: a(8), i
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  write(10) (a(i), i=1,8)
end subroutine

! Input: a retained subscript that is a dummy argument may be argument
! associated with an element being read, so it must not be collapsed.
! CHECK-LABEL: func @_QPread_dummy_subscript(
subroutine read_dummy_subscript(a, k, n)
  integer :: n
  integer :: a(10, n), k
  ! CHECK: fir.do_loop
  ! CHECK: fir.call @_FortranAioInputDescriptor
  read(10) (a(k, i), i=1,n)
end subroutine

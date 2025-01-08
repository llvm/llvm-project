! Test lowering of select rank to HLFIR
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module iface_helpers
interface
  subroutine r0(x)
    real :: x
  end subroutine
  subroutine r1(x)
    real :: x(:)
  end subroutine
  subroutine r2(x)
    real :: x(:, :)
  end subroutine
  subroutine r15(x)
    real :: x(:,:,:,:,:,:,:,:,:,:,:,:,:,:,:)
  end subroutine
  subroutine rdefault(x)
    real :: x(..)
  end subroutine
  subroutine rcdefault(x)
    character(*) :: x(..)
  end subroutine
  subroutine rassumed_size(x)
    real :: x(*)
  end subroutine

  subroutine ra0(x)
    real, allocatable  :: x
  end subroutine
  subroutine ra1(x)
    real, allocatable  :: x(:)
  end subroutine
  subroutine ra2(x)
    real, allocatable  :: x(:, :)
  end subroutine
  subroutine ra15(x)
    real, allocatable  :: x(:,:,:,:,:,:,:,:,:,:,:,:,:,:,:)
  end subroutine
  subroutine radefault(x)
    real, allocatable  :: x(..)
  end subroutine

  subroutine rup0(x)
    class(*) :: x
  end subroutine
  subroutine rup1(x)
    class(*)  :: x(:)
  end subroutine
  subroutine rupdefault(x)
    class(*) :: x(..)
  end subroutine

end interface
end module

subroutine test_single_case(x)
  use iface_helpers
  real :: x(..)
  select rank (x)
   rank (1)
    call r1(x)
  end select
end subroutine

subroutine test_simple_case(x)
  use iface_helpers
  real :: x(..)
  select rank (x)
   rank default
      call rdefault(x)
   rank (1)
      call r1(x)
   rank (15)
      call r15(x)
   rank (0)
      call r0(x)
  end select
end subroutine

subroutine test_rank_star(x)
  use iface_helpers
  real :: x(..)
  select rank (x)
   rank (2)
      call r2(x)
   rank default
      call rdefault(x)
   rank (1)
      call r1(x)
   rank (*)
      ! test no copy in/out is generated here.
      call rassumed_size(x)
  end select
end subroutine


subroutine test_renaming(x)
  use iface_helpers
  real :: x(..)
  select rank (new_name => x)
   rank (1)
    call r1(new_name)
    call rdefault(x)
  end select
end subroutine

subroutine test_no_case(x)
  real :: x(..)
  select rank (x)
  end select
end subroutine

subroutine test_rank_star_attributes(x)
  use iface_helpers
  real, optional, asynchronous, target :: x(..)
  ! The declare generated for the associating entity should have the
  ! TARGET and ASYNCHRONOUS attribute, but not the OPTIONAL attribute.
  ! F'2023 11.1.3.3 and 11.1.10.3.
  select rank (x)
   rank (2)
      call r2(x)
   rank default
      call rdefault(x)
   rank (*)
      call rassumed_size(x)
  end select
end subroutine

subroutine test_rank_star_contiguous(x)
  use iface_helpers
  real, target, contiguous :: x(..)
  ! Test simple hlfir.declare without fir.box are generated for
  ! ranked case and that non copy-in/out is generated when passing
  ! associating entity to implicit interfaces.
  select rank (x)
   rank (2)
    call r2_implicit(x)
   rank default
    ! TODO: hlfir.declare could be given the CONTIGUOUS attribute.
    call rdefault(x)
   rank (1)
    call r1_implicit(x)
   rank (*)
    call r1_implicit(x)
  end select
end subroutine

subroutine test_rank_star_contiguous_character(x, n)
  use iface_helpers
  integer(8) :: n
  character(n), contiguous :: x(..)
  select rank (x)
   ! test fir.box is properly converted to fir.boxchar in the hlfir.declare
   ! for the associating entity.
   rank (0)
    call rc0_implicit(x)
   rank default
    call rcdefault(x)
   rank (1)
    call rc1_implicit(x)
   rank (*)
    call rc1_implicit(x)
  end select
end subroutine

subroutine test_simple_alloc(x)
  use iface_helpers
  real, allocatable :: x(..)
  ! test no is_assumed_size if generated and that associating entity
  ! hlfir.declare has allocatable attrbute.
  select rank (x)
   rank (2)
    call ra2(x)
   rank (0)
    call ra0(x)
   rank default
    call radefault(x)
   rank (1)
    call ra1(x)
  end select
end subroutine

subroutine test_character_alloc(x)
  character(:), allocatable :: x(..)
  ! test hlfir.declare for associating entities do not not have
  ! explicit type parameters.
  select rank (x)
   rank default
   rank (1)
  end select
end subroutine

subroutine test_explicit_character_ptr(x, n)
  use iface_helpers
  integer(8) :: n
  character(n), allocatable :: x(..)
  ! test hlfir.declare for associating entities have
  ! explicit type parameters.
  select rank (x)
   rank default
   rank (0)
  end select
end subroutine

subroutine test_assumed_character_ptr(x)
  use iface_helpers
  character(*), allocatable :: x(..)
  ! test hlfir.declare for associating entities have
  ! explicit type parameters.
  select rank (x)
   rank default
   rank (0)
  end select
end subroutine

subroutine test_polymorphic(x)
  use iface_helpers
  class(*) :: x(..)
  select rank (x)
   rank (1)
     call rup1(x)
   rank default
     call rupdefault(x)
   rank (0)
     call rup0(x)
  end select
end subroutine

subroutine test_nested_select_rank(x1, x2)
  use iface_helpers
  real :: x1(..), x2(..)
  select rank(x1)
    rank(0)
      select rank(x2)
        rank(0)
          call r0(x1)
          call r0(x2)
        rank(1)
          call r0(x1)
          call r1(x2)
        rank default
          call r0(x1)
          call rdefault(x2)
      end select
    rank(1)
      select rank(x2)
        rank(0)
          call r1(x1)
          call r0(x2)
        rank(1)
          call r1(x1)
          call r1(x2)
        rank default
          call r1(x1)
          call rdefault(x2)
      end select
    rank default
      select rank(x2)
        rank(0)
          call rdefault(x1)
          call r0(x2)
        rank(1)
          call rdefault(x1)
          call r1(x2)
        rank default
          call rdefault(x1)
          call rdefault(x2)
      end select
  end select
end subroutine

subroutine test_branching(x)
! Note branching into a select rank, or between cases, is illegal
! and caught by semantics.
  use iface_helpers
  real :: x(..)
  logical, external :: leave_now
  logical, external :: jump
  select rank (x)
   rank default
    if (jump()) goto 1
    call one()
1   call rdefault(x)
   rank (1)
    if (leave_now()) goto 3
    call r1(x)
   rank (2)
  end select
3 call the_end()
end subroutine

! CHECK-LABEL:   func.func @_QPtest_single_case(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_single_caseEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_4:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_4]], ^bb3, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_5:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_5]] : i8 [#fir.point, %[[VAL_3]], ^bb2, unit, ^bb3]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFtest_single_caseEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPr1(%[[VAL_7]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb3
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_simple_case(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_simple_caseEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_4:.*]] = arith.constant 15 : i8
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_6:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_6]], ^bb1, ^bb2
! CHECK:         ^bb1:
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2]]#0 {uniq_name = "_QFtest_simple_caseEx"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPrdefault(%[[VAL_7]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb2:
! CHECK:           %[[VAL_8:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_8]] : i8 [#fir.point, %[[VAL_3]], ^bb3, #fir.point, %[[VAL_4]], ^bb4, #fir.point, %[[VAL_5]], ^bb5, unit, ^bb1]
! CHECK:         ^bb3:
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {uniq_name = "_QFtest_simple_caseEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPr1(%[[VAL_10]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb4:
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QFtest_simple_caseEx"} : (!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>) -> (!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>, !fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>)
! CHECK:           fir.call @_QPr15(%[[VAL_12]]#0) fastmath<contract> : (!fir.box<!fir.array<?x?x?x?x?x?x?x?x?x?x?x?x?x?x?xf32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb5:
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<f32>
! CHECK:           %[[VAL_14:.*]] = fir.box_addr %[[VAL_13]] : (!fir.box<f32>) -> !fir.ref<f32>
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] {uniq_name = "_QFtest_simple_caseEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           fir.call @_QPr0(%[[VAL_15]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb6:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_rank_star(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_rank_starEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i8
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_5:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_5]], ^bb5, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_6:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_6]] : i8 [#fir.point, %[[VAL_3]], ^bb2, #fir.point, %[[VAL_4]], ^bb4, unit, ^bb3]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFtest_rank_starEx"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           fir.call @_QPr2(%[[VAL_8]]#0) fastmath<contract> : (!fir.box<!fir.array<?x?xf32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb3:
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]]#0 {uniq_name = "_QFtest_rank_starEx"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPrdefault(%[[VAL_9]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb4:
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFtest_rank_starEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPr1(%[[VAL_11]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb5:
! CHECK:           %[[VAL_12:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_13:.*]] = fir.box_addr %[[VAL_2]]#1 : (!fir.box<!fir.array<*:f32>>) -> !fir.ref<!fir.array<*:f32>>
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.array<*:f32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_14]](%[[VAL_15]]) {uniq_name = "_QFtest_rank_starEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPrassumed_size(%[[VAL_16]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb6:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_renaming(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_renamingEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_4:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_4]], ^bb3, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_5:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_5]] : i8 [#fir.point, %[[VAL_3]], ^bb2, unit, ^bb3]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFtest_renamingEnew_name"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPr1(%[[VAL_7]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           fir.call @_QPrdefault(%[[VAL_2]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb3
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_no_case(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_no_caseEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_3]], ^bb2, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_4:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_4]] : i8 [unit, ^bb2]
! CHECK:         ^bb2:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_rank_star_attributes(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.asynchronous, fir.bindc_name = "x", fir.optional, fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<asynchronous, optional, target>, uniq_name = "_QFtest_rank_star_attributesEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i8
! CHECK:           %[[VAL_4:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_4]], ^bb4, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_5:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_5]] : i8 [#fir.point, %[[VAL_3]], ^bb2, unit, ^bb3]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {fortran_attrs = #fir.var_attrs<asynchronous, target>, uniq_name = "_QFtest_rank_star_attributesEx"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           fir.call @_QPr2(%[[VAL_7]]#0) fastmath<contract> : (!fir.box<!fir.array<?x?xf32>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb3:
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_2]]#0 {fortran_attrs = #fir.var_attrs<asynchronous, target>, uniq_name = "_QFtest_rank_star_attributesEx"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPrdefault(%[[VAL_8]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb4:
! CHECK:           %[[VAL_9:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_10:.*]] = fir.box_addr %[[VAL_2]]#1 : (!fir.box<!fir.array<*:f32>>) -> !fir.ref<!fir.array<*:f32>>
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.array<*:f32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_12]]) {fortran_attrs = #fir.var_attrs<asynchronous, target>, uniq_name = "_QFtest_rank_star_attributesEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPrassumed_size(%[[VAL_13]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb5:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_rank_star_contiguous(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x", fir.contiguous, fir.target}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<contiguous, target>, uniq_name = "_QFtest_rank_star_contiguousEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i8
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_5:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_5]], ^bb5, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_6:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_6]] : i8 [#fir.point, %[[VAL_3]], ^bb2, #fir.point, %[[VAL_4]], ^bb4, unit, ^bb3]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_9]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_11]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_10]]#1, %[[VAL_12]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_13]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest_rank_star_contiguousEx"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
! CHECK:           fir.call @_QPr2_implicit(%[[VAL_14]]#1) fastmath<contract> : (!fir.ref<!fir.array<?x?xf32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb3:
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_2]]#0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest_rank_star_contiguousEx"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPrdefault(%[[VAL_15]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb4:
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[VAL_18:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_19:.*]]:3 = fir.box_dims %[[VAL_16]], %[[VAL_18]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_20:.*]] = fir.shape %[[VAL_19]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_17]](%[[VAL_20]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest_rank_star_contiguousEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPr1_implicit(%[[VAL_21]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb5:
! CHECK:           %[[VAL_22:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_23:.*]] = fir.box_addr %[[VAL_2]]#1 : (!fir.box<!fir.array<*:f32>>) -> !fir.ref<!fir.array<*:f32>>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<!fir.array<*:f32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[VAL_25:.*]] = fir.shape %[[VAL_22]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_24]](%[[VAL_25]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest_rank_star_contiguousEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPr1_implicit(%[[VAL_26]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb6:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_rank_star_contiguous_character(
! CHECK-SAME:                                                      %[[VAL_0:.*]]: !fir.box<!fir.array<*:!fir.char<1,?>>> {fir.bindc_name = "x", fir.contiguous},
! CHECK-SAME:                                                      %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_rank_star_contiguous_characterEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_7]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest_rank_star_contiguous_characterEx"} : (!fir.box<!fir.array<*:!fir.char<1,?>>>, i64, !fir.dscope) -> (!fir.box<!fir.array<*:!fir.char<1,?>>>, !fir.box<!fir.array<*:!fir.char<1,?>>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_11:.*]] = fir.is_assumed_size %[[VAL_8]]#0 : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> i1
! CHECK:           cf.cond_br %[[VAL_11]], ^bb5, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_12:.*]] = fir.box_rank %[[VAL_8]]#0 : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> i8
! CHECK:           fir.select_case %[[VAL_12]] : i8 [#fir.point, %[[VAL_9]], ^bb2, #fir.point, %[[VAL_10]], ^bb4, unit, ^bb3]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> !fir.box<!fir.char<1,?>>
! CHECK:           %[[VAL_14:.*]] = fir.box_addr %[[VAL_13]] : (!fir.box<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_15:.*]] = fir.box_elesize %[[VAL_13]] : (!fir.box<!fir.char<1,?>>) -> index
! CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_14]] typeparams %[[VAL_15]] {uniq_name = "_QFtest_rank_star_contiguous_characterEx"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:           fir.call @_QPrc0_implicit(%[[VAL_16]]#0) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb3:
! CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_8]]#0 typeparams %[[VAL_7]] {uniq_name = "_QFtest_rank_star_contiguous_characterEx"} : (!fir.box<!fir.array<*:!fir.char<1,?>>>, i64) -> (!fir.box<!fir.array<*:!fir.char<1,?>>>, !fir.box<!fir.array<*:!fir.char<1,?>>>)
! CHECK:           fir.call @_QPrcdefault(%[[VAL_17]]#0) fastmath<contract> : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb4:
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:           %[[VAL_19:.*]] = fir.box_addr %[[VAL_18]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_18]], %[[VAL_20]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_22:.*]] = fir.box_elesize %[[VAL_18]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_23:.*]] = fir.shape %[[VAL_21]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_19]](%[[VAL_23]]) typeparams %[[VAL_22]] {uniq_name = "_QFtest_rank_star_contiguous_characterEx"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]]#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_26:.*]] = fir.emboxchar %[[VAL_25]], %[[VAL_22]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPrc1_implicit(%[[VAL_26]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb5:
! CHECK:           %[[VAL_27:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_28:.*]] = fir.box_addr %[[VAL_8]]#1 : (!fir.box<!fir.array<*:!fir.char<1,?>>>) -> !fir.ref<!fir.array<*:!fir.char<1,?>>>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.array<*:!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:           %[[VAL_30:.*]] = fir.shape %[[VAL_27]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_31:.*]]:2 = hlfir.declare %[[VAL_29]](%[[VAL_30]]) typeparams %[[VAL_7]] {uniq_name = "_QFtest_rank_star_contiguous_characterEx"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i64) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]]#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_33:.*]] = fir.emboxchar %[[VAL_32]], %[[VAL_7]] : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPrc1_implicit(%[[VAL_33]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:           cf.br ^bb6
! CHECK:         ^bb6:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_simple_alloc(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_simple_allocEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i8
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i8
! CHECK-NOT: fir.is_assumed_size
! CHECK:           %[[VAL_6:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> i8
! CHECK:           fir.select_case %[[VAL_6]] : i8 [#fir.point, %[[VAL_3]], ^bb1, #fir.point, %[[VAL_4]], ^bb2, #fir.point, %[[VAL_5]], ^bb4, unit, ^bb3]
! CHECK:         ^bb1:
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_simple_allocEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>)
! CHECK:           fir.call @_QPra2(%[[VAL_8]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb2:
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_simple_allocEx"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK:           fir.call @_QPra0(%[[VAL_10]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb3:
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_2]]#0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_simple_allocEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>)
! CHECK:           fir.call @_QPradefault(%[[VAL_11]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb4:
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_simple_allocEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK:           fir.call @_QPra1(%[[VAL_13]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb5:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_character_alloc(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_character_allocEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_4:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>) -> i8
! CHECK:           fir.select_case %[[VAL_4]] : i8 [#fir.point, %[[VAL_3]], ^bb2, unit, ^bb1]
! CHECK:         ^bb1:
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]]#0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_character_allocEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>)
! CHECK:           cf.br ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>
! CHECK:           %[[VAL_8:.*]] = fir.box_elesize %[[VAL_7]] : (!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>) -> index
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_6]] typeparams %[[VAL_8]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_character_allocEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>)
! CHECK:           cf.br ^bb3
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_explicit_character_ptr(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>> {fir.bindc_name = "x"},
! CHECK-SAME:                                              %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_explicit_character_ptrEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_7]] dummy_scope %[[VAL_2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_explicit_character_ptrEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, i64, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_10:.*]] = fir.box_rank %[[VAL_8]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>) -> i8
! CHECK:           fir.select_case %[[VAL_10]] : i8 [#fir.point, %[[VAL_9]], ^bb2, unit, ^bb1]
! CHECK:         ^bb1:
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_8]]#0 typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_explicit_character_ptrEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, i64) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>)
! CHECK:           cf.br ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_explicit_character_ptrEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, i64) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:           cf.br ^bb3
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_assumed_character_ptr(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>
! CHECK:           %[[VAL_3:.*]] = fir.box_elesize %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>) -> index
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_3]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_assumed_character_ptrEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, index, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_6:.*]] = fir.box_rank %[[VAL_4]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>) -> i8
! CHECK:           fir.select_case %[[VAL_6]] : i8 [#fir.point, %[[VAL_5]], ^bb2, unit, ^bb1]
! CHECK:         ^bb1:
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_assumed_character_ptrEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>)
! CHECK:           cf.br ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_assumed_character_ptrEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, index) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:           cf.br ^bb3
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_polymorphic(
! CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_polymorphicEx"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_5:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.class<!fir.array<*:none>>) -> i1
! CHECK:           cf.cond_br %[[VAL_5]], ^bb3, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_6:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.class<!fir.array<*:none>>) -> i8
! CHECK:           fir.select_case %[[VAL_6]] : i8 [#fir.point, %[[VAL_3]], ^bb2, #fir.point, %[[VAL_4]], ^bb4, unit, ^bb3]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.class<!fir.array<*:none>>) -> !fir.class<!fir.array<?xnone>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFtest_polymorphicEx"} : (!fir.class<!fir.array<?xnone>>) -> (!fir.class<!fir.array<?xnone>>, !fir.class<!fir.array<?xnone>>)
! CHECK:           fir.call @_QPrup1(%[[VAL_8]]#0) fastmath<contract> : (!fir.class<!fir.array<?xnone>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb3:
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]]#0 {uniq_name = "_QFtest_polymorphicEx"} : (!fir.class<!fir.array<*:none>>) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           fir.call @_QPrupdefault(%[[VAL_9]]#0) fastmath<contract> : (!fir.class<!fir.array<*:none>>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb4:
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.class<!fir.array<*:none>>) -> !fir.class<none>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFtest_polymorphicEx"} : (!fir.class<none>) -> (!fir.class<none>, !fir.class<none>)
! CHECK:           fir.call @_QPrup0(%[[VAL_11]]#0) fastmath<contract> : (!fir.class<none>) -> ()
! CHECK:           cf.br ^bb5
! CHECK:         ^bb5:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_nested_select_rank(
! CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x1"},
! CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x2"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_nested_select_rankEx1"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_7:.*]] = fir.is_assumed_size %[[VAL_3]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_7]], ^bb14, ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_8:.*]] = fir.box_rank %[[VAL_3]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_8]] : i8 [#fir.point, %[[VAL_5]], ^bb2, #fir.point, %[[VAL_6]], ^bb8, unit, ^bb14]
! CHECK:         ^bb2:
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<f32>
! CHECK:           %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<f32>) -> !fir.ref<f32>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFtest_nested_select_rankEx1"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_13:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_14:.*]] = fir.is_assumed_size %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_14]], ^bb6, ^bb3
! CHECK:         ^bb3:
! CHECK:           %[[VAL_15:.*]] = fir.box_rank %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_15]] : i8 [#fir.point, %[[VAL_12]], ^bb4, #fir.point, %[[VAL_13]], ^bb5, unit, ^bb6]
! CHECK:         ^bb4:
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<f32>
! CHECK:           %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<f32>) -> !fir.ref<f32>
! CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           fir.call @_QPr0(%[[VAL_11]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:           fir.call @_QPr0(%[[VAL_18]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:           cf.br ^bb7
! CHECK:         ^bb5:
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_19]] {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPr0(%[[VAL_11]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:           fir.call @_QPr1(%[[VAL_20]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb7
! CHECK:         ^bb6:
! CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_4]]#0 {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPr0(%[[VAL_11]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:           fir.call @_QPrdefault(%[[VAL_21]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb7
! CHECK:         ^bb7:
! CHECK:           cf.br ^bb20
! CHECK:         ^bb8:
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_22]] {uniq_name = "_QFtest_nested_select_rankEx1"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_24:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_25:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_26:.*]] = fir.is_assumed_size %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_26]], ^bb12, ^bb9
! CHECK:         ^bb9:
! CHECK:           %[[VAL_27:.*]] = fir.box_rank %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_27]] : i8 [#fir.point, %[[VAL_24]], ^bb10, #fir.point, %[[VAL_25]], ^bb11, unit, ^bb12]
! CHECK:         ^bb10:
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<f32>
! CHECK:           %[[VAL_29:.*]] = fir.box_addr %[[VAL_28]] : (!fir.box<f32>) -> !fir.ref<f32>
! CHECK:           %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_29]] {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           fir.call @_QPr1(%[[VAL_23]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           fir.call @_QPr0(%[[VAL_30]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:           cf.br ^bb13
! CHECK:         ^bb11:
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_32:.*]]:2 = hlfir.declare %[[VAL_31]] {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPr1(%[[VAL_23]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           fir.call @_QPr1(%[[VAL_32]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb13
! CHECK:         ^bb12:
! CHECK:           %[[VAL_33:.*]]:2 = hlfir.declare %[[VAL_4]]#0 {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPr1(%[[VAL_23]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           fir.call @_QPrdefault(%[[VAL_33]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb13
! CHECK:         ^bb13:
! CHECK:           cf.br ^bb20
! CHECK:         ^bb14:
! CHECK:           %[[VAL_34:.*]]:2 = hlfir.declare %[[VAL_3]]#0 {uniq_name = "_QFtest_nested_select_rankEx1"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_35:.*]] = arith.constant 0 : i8
! CHECK:           %[[VAL_36:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_37:.*]] = fir.is_assumed_size %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_37]], ^bb18, ^bb15
! CHECK:         ^bb15:
! CHECK:           %[[VAL_38:.*]] = fir.box_rank %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_38]] : i8 [#fir.point, %[[VAL_35]], ^bb16, #fir.point, %[[VAL_36]], ^bb17, unit, ^bb18]
! CHECK:         ^bb16:
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<f32>
! CHECK:           %[[VAL_40:.*]] = fir.box_addr %[[VAL_39]] : (!fir.box<f32>) -> !fir.ref<f32>
! CHECK:           %[[VAL_41:.*]]:2 = hlfir.declare %[[VAL_40]] {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           fir.call @_QPrdefault(%[[VAL_34]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           fir.call @_QPr0(%[[VAL_41]]#1) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:           cf.br ^bb19
! CHECK:         ^bb17:
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_43:.*]]:2 = hlfir.declare %[[VAL_42]] {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           fir.call @_QPrdefault(%[[VAL_34]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           fir.call @_QPr1(%[[VAL_43]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb19
! CHECK:         ^bb18:
! CHECK:           %[[VAL_44:.*]]:2 = hlfir.declare %[[VAL_4]]#0 {uniq_name = "_QFtest_nested_select_rankEx2"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPrdefault(%[[VAL_34]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           fir.call @_QPrdefault(%[[VAL_44]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb19
! CHECK:         ^bb19:
! CHECK:           cf.br ^bb20
! CHECK:         ^bb20:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_branching(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_branchingEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i8
! CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i8
! CHECK:           %[[VAL_5:.*]] = fir.is_assumed_size %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i1
! CHECK:           cf.cond_br %[[VAL_5]], ^bb1, ^bb2
! CHECK:         ^bb1:
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_2]]#0 {uniq_name = "_QFtest_branchingEx"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_7:.*]] = fir.call @_QPjump() fastmath<contract> : () -> !fir.logical<4>
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.logical<4>) -> i1
! CHECK:           %[[VAL_9:.*]] = arith.constant true
! CHECK:           %[[VAL_10:.*]] = arith.xori %[[VAL_8]], %[[VAL_9]] : i1
! CHECK:           fir.if %[[VAL_10]] {
! CHECK:             fir.call @_QPone() fastmath<contract> : () -> ()
! CHECK:           }
! CHECK:           fir.call @_QPrdefault(%[[VAL_6]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           cf.br ^bb7
! CHECK:         ^bb2:
! CHECK:           %[[VAL_11:.*]] = fir.box_rank %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> i8
! CHECK:           fir.select_case %[[VAL_11]] : i8 [#fir.point, %[[VAL_3]], ^bb3, #fir.point, %[[VAL_4]], ^bb6, unit, ^bb1]
! CHECK:         ^bb3:
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFtest_branchingEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %[[VAL_14:.*]] = fir.call @_QPleave_now() fastmath<contract> : () -> !fir.logical<4>
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
! CHECK:           cf.cond_br %[[VAL_15]], ^bb4, ^bb5
! CHECK:         ^bb4:
! CHECK:           cf.br ^bb8
! CHECK:         ^bb5:
! CHECK:           fir.call @_QPr1(%[[VAL_13]]#0) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> ()
! CHECK:           cf.br ^bb7
! CHECK:         ^bb6:
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] {uniq_name = "_QFtest_branchingEx"} : (!fir.box<!fir.array<?x?xf32>>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
! CHECK:           cf.br ^bb7
! CHECK:         ^bb7:
! CHECK:           cf.br ^bb8
! CHECK:         ^bb8:
! CHECK:           fir.call @_QPthe_end() fastmath<contract> : () -> ()
! CHECK:           return
! CHECK:         }

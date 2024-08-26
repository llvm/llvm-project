! Test assumed-rank capture inside internal procedures.
! RUN: bbc -emit-hlfir -o - %s -allow-assumed-rank | FileCheck %s

subroutine test_assumed_rank(x)
  real :: x(..)
interface
subroutine some_sub(x)
  real :: x(..)
end subroutine
end interface
  call internal()
contains
subroutine internal()
  call some_sub(x)
end subroutine
end subroutine
! CHECK-LABEL:   func.func @_QPtest_assumed_rank(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_assumed_rankEx"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca tuple<!fir.box<!fir.array<*:f32>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_4]] : (!fir.ref<tuple<!fir.box<!fir.array<*:f32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<*:f32>>>
! CHECK:           %[[VAL_6:.*]] = fir.rebox_assumed_rank %[[VAL_2]]#0 lbs preserve : (!fir.box<!fir.array<*:f32>>) -> !fir.box<!fir.array<*:f32>>
! CHECK:           fir.store %[[VAL_6]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.array<*:f32>>>
! CHECK:           fir.call @_QFtest_assumed_rankPinternal(%[[VAL_3]]) fastmath<contract> : (!fir.ref<tuple<!fir.box<!fir.array<*:f32>>>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFtest_assumed_rankPinternal(
! CHECK-SAME:                                                     %[[VAL_0:.*]]: !fir.ref<tuple<!fir.box<!fir.array<*:f32>>>> {fir.host_assoc})
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<tuple<!fir.box<!fir.array<*:f32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<*:f32>>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.array<*:f32>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFtest_assumed_rankEx"} : (!fir.box<!fir.array<*:f32>>) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
! CHECK:           fir.call @_QPsome_sub(%[[VAL_5]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
! CHECK:           return
! CHECK:         }


subroutine test_assumed_rank_optional(x)
  class(*), optional :: x(..)
interface
subroutine some_sub2(x)
  class(*) :: x(..)
end subroutine
end interface
  call internal()
contains
subroutine internal()
  call some_sub2(x)
end subroutine
end subroutine
! CHECK-LABEL:   func.func @_QPtest_assumed_rank_optional(
! CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.class<!fir.array<*:none>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest_assumed_rank_optionalEx"} : (!fir.class<!fir.array<*:none>>, !fir.dscope) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca tuple<!fir.class<!fir.array<*:none>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_4]] : (!fir.ref<tuple<!fir.class<!fir.array<*:none>>>>, i32) -> !fir.ref<!fir.class<!fir.array<*:none>>>
! CHECK:           %[[VAL_6:.*]] = fir.is_present %[[VAL_2]]#0 : (!fir.class<!fir.array<*:none>>) -> i1
! CHECK:           fir.if %[[VAL_6]] {
! CHECK:             %[[VAL_7:.*]] = fir.rebox_assumed_rank %[[VAL_2]]#0 lbs preserve : (!fir.class<!fir.array<*:none>>) -> !fir.class<!fir.array<*:none>>
! CHECK:             fir.store %[[VAL_7]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.array<*:none>>>
! CHECK:           } else {
! CHECK:             %[[VAL_8:.*]] = fir.zero_bits !fir.ref<none>
! CHECK:             %[[VAL_9:.*]] = fir.embox %[[VAL_8]] : (!fir.ref<none>) -> !fir.class<none>
! CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.class<none>) -> !fir.class<!fir.array<*:none>>
! CHECK:             fir.store %[[VAL_10]] to %[[VAL_5]] : !fir.ref<!fir.class<!fir.array<*:none>>>
! CHECK:           }
! CHECK:           fir.call @_QFtest_assumed_rank_optionalPinternal(%[[VAL_3]]) fastmath<contract> : (!fir.ref<tuple<!fir.class<!fir.array<*:none>>>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFtest_assumed_rank_optionalPinternal(
! CHECK-SAME:                                                              %[[VAL_0:.*]]: !fir.ref<tuple<!fir.class<!fir.array<*:none>>>> {fir.host_assoc})
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<tuple<!fir.class<!fir.array<*:none>>>>, i32) -> !fir.ref<!fir.class<!fir.array<*:none>>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.class<!fir.array<*:none>>>
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.class<!fir.array<*:none>>) -> !fir.ref<!fir.array<*:none>>
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.array<*:none>>) -> i64
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:           %[[VAL_9:.*]] = fir.absent !fir.class<!fir.array<*:none>>
! CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_8]], %[[VAL_4]], %[[VAL_9]] : !fir.class<!fir.array<*:none>>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]] {fortran_attrs = #fir.var_attrs<optional, host_assoc>, uniq_name = "_QFtest_assumed_rank_optionalEx"} : (!fir.class<!fir.array<*:none>>) -> (!fir.class<!fir.array<*:none>>, !fir.class<!fir.array<*:none>>)
! CHECK:           fir.call @_QPsome_sub2(%[[VAL_11]]#0) fastmath<contract> : (!fir.class<!fir.array<*:none>>) -> ()
! CHECK:           return
! CHECK:         }


subroutine test_assumed_rank_ptr(x)
  real, pointer :: x(..)
interface
subroutine some_sub3(x)
  real, pointer :: x(..)
end subroutine
end interface
  call internal()
contains
subroutine internal()
  call some_sub3(x)
end subroutine
end subroutine
! CHECK-LABEL:   func.func @_QPtest_assumed_rank_ptr(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_assumed_rank_ptrEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_4]] : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>
! CHECK:           fir.store %[[VAL_2]]#0 to %[[VAL_5]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>
! CHECK:           fir.call @_QFtest_assumed_rank_ptrPinternal(%[[VAL_3]]) fastmath<contract> : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFtest_assumed_rank_ptrPinternal(
! CHECK-SAME:                                                         %[[VAL_0:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>> {fir.host_assoc})
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<pointer, host_assoc>, uniq_name = "_QFtest_assumed_rank_ptrEx"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>)
! CHECK:           fir.call @_QPsome_sub3(%[[VAL_5]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<*:f32>>>>) -> ()
! CHECK:           return
! CHECK:         }

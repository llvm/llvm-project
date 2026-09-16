! Test passing pointers results to pointer dummy arguments
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module presults
  interface
    subroutine bar_scalar(x)
      real, pointer :: x
    end subroutine
    subroutine bar(x)
      real, pointer :: x(:, :)
    end subroutine
    function get_scalar_pointer()
      real, pointer :: get_scalar_pointer
    end function
    function get_pointer()
      real, pointer :: get_pointer(:, :)
    end function
  end interface
  real, pointer :: x
  real, pointer :: xa(:, :)
contains

! CHECK-LABEL: test_scalar_null
subroutine test_scalar_null()
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<f32>>
! CHECK: %[[VAL_1:.*]] = fir.zero_bits !fir.ptr<f32>
! CHECK: %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK: fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: fir.call @_QPbar_scalar(%[[VAL_0]]) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> ()
  call bar_scalar(null())
end subroutine

! CHECK-LABEL: test_scalar_null_mold
subroutine test_scalar_null_mold()
! CHECK: %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<f32>>
! CHECK: %[[VAL_4:.*]] = fir.zero_bits !fir.ptr<f32>
! CHECK: %[[VAL_5:.*]] = fir.embox %[[VAL_4]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK: fir.store %[[VAL_5]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[VAL_TMP:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = ".tmp.intrinsic_result"}
! CHECK: fir.call @_QPbar_scalar(%[[VAL_TMP]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> ()
  call bar_scalar(null(x))
end subroutine

! CHECK-LABEL: test_scalar_result
subroutine test_scalar_result()
! CHECK: %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = ".result"}
! CHECK: %[[VAL_7:.*]] = fir.call @_QPget_scalar_pointer() {{.*}}: () -> !fir.box<!fir.ptr<f32>>
! CHECK: fir.save_result %[[VAL_7]] to %[[VAL_6]] : !fir.box<!fir.ptr<f32>>, !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[VAL_RES:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = ".tmp.func_result"}
! CHECK: fir.call @_QPbar_scalar(%[[VAL_RES]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> ()
  call bar_scalar(get_scalar_pointer())
end subroutine

! CHECK-LABEL: test_null
subroutine test_null()
! CHECK: %[[VAL_9:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK: %[[VAL_10:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x?xf32>>
! CHECK: %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK: %[[VAL_11:.*]] = fir.shape %[[VAL_8]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! CHECK: %[[VAL_12:.*]] = fir.embox %[[VAL_10]](%[[VAL_11]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK: fir.store %[[VAL_12]] to %[[VAL_9]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK: fir.call @_QPbar(%[[VAL_9]]) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> ()
  call bar(null())
end subroutine

! CHECK-LABEL: test_null_mold
subroutine test_null_mold()
! CHECK: %[[VAL_14:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK: %[[VAL_15:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x?xf32>>
! CHECK: %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK: %[[VAL_16:.*]] = fir.shape %[[VAL_13]], %[[VAL_13]] : (index, index) -> !fir.shape<2>
! CHECK: %[[VAL_17:.*]] = fir.embox %[[VAL_15]](%[[VAL_16]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK: fir.store %[[VAL_17]] to %[[VAL_14]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK: %[[VAL_TMP2:.*]]:2 = hlfir.declare %[[VAL_14]] {uniq_name = ".tmp.intrinsic_result"}
! CHECK: fir.call @_QPbar(%[[VAL_TMP2]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> ()
  call bar(null(xa))
end subroutine

! CHECK-LABEL: test_result
subroutine test_result()
! CHECK: %[[VAL_18:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>> {bindc_name = ".result"}
! CHECK: %[[VAL_19:.*]] = fir.call @_QPget_pointer() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK: fir.save_result %[[VAL_19]] to %[[VAL_18]] : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK: %[[VAL_RES2:.*]]:2 = hlfir.declare %[[VAL_18]] {uniq_name = ".tmp.func_result"}
! CHECK: fir.call @_QPbar(%[[VAL_RES2]]#0) {{.*}}: (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>) -> ()
  call bar(get_pointer())
end subroutine

end module

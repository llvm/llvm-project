! Test lowering of elemental subroutine calls with array arguments
! RUN: bbc -o - -emit-fir -hlfir=false %s | FileCheck %s

! CHECK-LABEL: func @_QPtest_elem_sub(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_3:.*]]: !fir.ref<complex<f32>>{{.*}}) {
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant -1 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = fir.slice %[[VAL_8]], %[[VAL_12]], %[[VAL_10]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_3]] : !fir.ref<complex<f32>>
! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_17:.*]] = arith.subi %[[VAL_6]]#1, %[[VAL_15]] : index
! CHECK:         fir.do_loop %[[VAL_18:.*]] = %[[VAL_16]] to %[[VAL_17]] step %[[VAL_15]] {
! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_18]], %[[VAL_19]] : index
! CHECK:           %[[VAL_21:.*]] = fir.array_coor %[[VAL_0]] %[[VAL_20]] : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:           %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_23:.*]] = arith.addi %[[VAL_18]], %[[VAL_22]] : index
! CHECK:           %[[VAL_24:.*]] = fir.array_coor %[[VAL_1]] {{\[}}%[[VAL_13]]] %[[VAL_23]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_25:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_26:.*]] = fir.emboxchar %[[VAL_24]], %[[VAL_25]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           fir.call @_QPfoo(%[[VAL_21]], %[[VAL_26]], %[[VAL_2]], %[[VAL_14]]) {{.*}}: (!fir.ref<f32>, !fir.boxchar<1>, !fir.ref<i32>, complex<f32>) -> ()
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine test_elem_sub(x, c, i, z)
  real :: x(:)
  character(*) :: c(:)
  integer :: i
  complex :: z
  interface
    elemental subroutine foo(x, c, i, z)
      real, intent(out) :: x
      character(*), intent(inout) :: c
      integer, intent(in) :: i
      complex, value :: z
    end subroutine
  end interface

  call foo(x, c(10:1:-1), i, z)
end subroutine

! CHECK-LABEL: func @_QPtest_elem_sub_no_array_args(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}) {
subroutine test_elem_sub_no_array_args(i, j)
  integer :: i, j
  interface
    elemental subroutine bar(i, j)
      integer, intent(out) :: i
      integer, intent(in) :: j
    end subroutine
  end interface
  call bar(i, j)
  ! CHECK:         fir.call @_QPbar(%[[VAL_0]], %[[VAL_1]]) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
end subroutine

! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test_co_sum
  integer :: i, array_i(2), status
  real :: r, array_r(2)
  double precision :: d, array_d(2)
  character(len=1) :: message

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_I:.*]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: mif.co_sum %[[V1]] : (!fir.box<i32>)
  call co_sum(i)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_D:.*]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  ! CHECK: mif.co_sum %[[V1]] : (!fir.box<f64>)
  call co_sum(d)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_R:.*]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: mif.co_sum %[[V1]] : (!fir.box<f32>)
  call co_sum(r)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_I]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: mif.co_sum %[[V1]] result %[[C1_i32:.*]] : (!fir.box<i32>, i32)
  call co_sum(i,       result_image=1)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_D]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE:.*]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_sum %[[V1]] result %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 errmsg %[[V2]] : (!fir.box<f64>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_sum(d,       result_image=1, stat=status, errmsg=message)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_R]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_sum %[[V1]] result %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 errmsg %[[V2]] : (!fir.box<f32>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_sum(r,       result_image=1, stat=status, errmsg=message)

  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_I:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
  ! CHECK: mif.co_sum %[[V1]] : (!fir.box<!fir.array<2xi32>>)
  call co_sum(array_i)
   
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_D:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf64>>
  ! CHECK: mif.co_sum %[[V1]] result %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 : (!fir.box<!fir.array<2xf64>>, i32, !fir.ref<i32>)
  call co_sum(array_d, result_image=1, stat=status)

  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_R:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_sum %[[V1]] result %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 errmsg %[[V2]] : (!fir.box<!fir.array<2xf32>>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_sum(array_r, result_image=1, stat= status, errmsg=message)

end program

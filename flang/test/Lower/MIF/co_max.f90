! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test_co_max
  integer :: i, array_i(2), status
  real :: r, array_r(2)
  double precision :: d, array_d(2)
  character(len=1) :: c, array_c(2), message

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_I:.*]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: mif.co_max %[[V1]] : (!fir.box<i32>)
  call co_max(i)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_C:.*]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_max %[[V1]] : (!fir.box<!fir.char<1>>)
  call co_max(c)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_D:.*]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  ! CHECK: mif.co_max %[[V1]] : (!fir.box<f64>)
  call co_max(d)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_R:.*]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: mif.co_max %[[V1]] : (!fir.box<f32>)
  call co_max(r)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_I]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: mif.co_max %[[V1]] result %[[C1_i32:.*]] : (!fir.box<i32>, i32)
  call co_max(i,       result_image=1)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_D]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE:.*]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_max %[[V1]] result %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 errmsg %[[V2]] : (!fir.box<f64>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_max(d,       result_image=1, stat=status, errmsg=message)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_R]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_max %[[V1]] result %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 errmsg %[[V2]] : (!fir.box<f32>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_max(r,       result_image=1, stat=status, errmsg=message)

  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_I:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
  ! CHECK: mif.co_max %[[V1]] : (!fir.box<!fir.array<2xi32>>)
  call co_max(array_i)
   
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_C:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1>>>
  ! CHECK: mif.co_max %[[V1]] result %[[C1_i32:.*]] : (!fir.box<!fir.array<2x!fir.char<1>>>, i32)
  call co_max(array_c, result_image=1) 
   
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_D:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf64>>
  ! CHECK: mif.co_max %[[V1]] result %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 : (!fir.box<!fir.array<2xf64>>, i32, !fir.ref<i32>)
  call co_max(array_d, result_image=1, stat=status)

  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_R:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_max %[[V1]] result %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 errmsg %[[V2]] : (!fir.box<!fir.array<2xf32>>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_max(array_r, result_image=1, stat= status, errmsg=message)

end program

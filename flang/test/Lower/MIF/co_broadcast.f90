! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test_co_broadcast
  integer :: i, array_i(2), status
  real :: r, array_r(2)
  double precision :: d, array_d(2)
  complex :: c, array_c(2)
  character(len=1) :: message

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_I:.*]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: mif.co_broadcast %[[V1]] source %[[C1_i32:.*]] : (!fir.box<i32>, i32)
  call co_broadcast(i,       source_image=1)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_C:.*]]#0 : (!fir.ref<complex<f32>>) -> !fir.box<complex<f32>>
  ! CHECK: mif.co_broadcast %[[V1]] source %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 : (!fir.box<complex<f32>>,  i32, !fir.ref<i32>)
  call co_broadcast(c,       source_image=1, stat=status)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_D:.*]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE:.*]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_broadcast %[[V1]] source %[[C1_i32:.*]] stat %[[STATUS]]#0 errmsg %[[V2]] : (!fir.box<f64>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_broadcast(d,       source_image=1, stat=status, errmsg=message)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_R:.*]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_broadcast %[[V1]] source %[[C1_i32:.*]] stat %[[STATUS]]#0 errmsg %[[V2]] : (!fir.box<f32>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_broadcast(r,       source_image=1, stat=status, errmsg=message)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_I:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
  ! CHECK: mif.co_broadcast %[[V1]] source %[[C1_i32:.*]] : (!fir.box<!fir.array<2xi32>>, i32)
  call co_broadcast(array_i, source_image=1)
  
  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_C:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.array<2xcomplex<f32>>>
  ! CHECK: mif.co_broadcast %[[V1]] source %[[C1_i32:.*]] : (!fir.box<!fir.array<2xcomplex<f32>>>, i32)
  call co_broadcast(array_c, source_image=1)
  
  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_D:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf64>>
  ! CHECK: mif.co_broadcast %[[V1]] source %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 : (!fir.box<!fir.array<2xf64>>, i32, !fir.ref<i32>)
  call co_broadcast(array_d, source_image=1, stat=status)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_C:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: mif.co_broadcast %[[V1]] source %[[C1_i32:.*]] stat %[[STATUS:.*]]#0 errmsg %[[V2]] : (!fir.box<!fir.array<2xf32>>, i32, !fir.ref<i32>, !fir.box<!fir.char<1>>)
  call co_broadcast(array_r, source_image=1, stat= status, errmsg=message)

end program

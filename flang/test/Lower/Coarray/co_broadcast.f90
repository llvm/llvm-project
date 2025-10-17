! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test_co_broadcast
  integer :: i, array_i(2), status
  real :: r, array_r(2)
  double precision :: d, array_d(2)
  complex :: c, array_c(2)
  character(len=1) :: message

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_I:.*]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_broadcast(%[[V5]], %[[IMAGE_RESULT]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_broadcast(i,       source_image=1)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_C:.*]]#0 : (!fir.ref<complex<f32>>) -> !fir.box<complex<f32>>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<complex<f32>>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_broadcast(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS:.*]], %[[V2]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_broadcast(c,       source_image=1, stat=status)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_D:.*]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE:.*]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<f64>) -> !fir.box<none>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  ! CHECK: fir.call @_QMprifPprif_co_broadcast(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V5]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_broadcast(d,       source_image=1, stat=status, errmsg=message)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_R:.*]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<f32>) -> !fir.box<none>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  ! CHECK: fir.call @_QMprifPprif_co_broadcast(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V5]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_broadcast(r,       source_image=1, stat=status, errmsg=message)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_I:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_broadcast(%[[V5]], %[[IMAGE_RESULT]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_broadcast(array_i, source_image=1)
  
  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_C:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.array<2xcomplex<f32>>>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xcomplex<f32>>>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_broadcast(%[[V5]], %[[IMAGE_RESULT]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_broadcast(array_c, source_image=1)
  
  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_D:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf64>>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xf64>>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_broadcast(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V2]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_broadcast(array_d, source_image=1, stat=status)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_C:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<none>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  ! CHECK: fir.call @_QMprifPprif_co_broadcast(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V5]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_broadcast(array_r, source_image=1, stat= status, errmsg=message)

end program

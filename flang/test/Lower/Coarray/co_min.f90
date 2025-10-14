! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

program test_co_min
  integer :: i, array_i(2), status
  real :: r, array_r(2)
  double precision :: d, array_d(2)
  character(len=1) :: c, array_c(2), message

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_I:.*]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V5]], %[[V2]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(i)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_C:.*]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.char<1>>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_min_character(%[[V5]], %[[V2]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(c)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_D:.*]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<f64>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V5]], %[[V2]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(d)

  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_R:.*]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<f32>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V5]], %[[V2]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(r)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_I]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V5]], %[[IMAGE_RESULT]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(i,       result_image=1)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_D]]#0 : (!fir.ref<f64>) -> !fir.box<f64>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE:.*]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<f64>) -> !fir.box<none>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS:.*]], %[[V5]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(d,       result_image=1, stat=status, errmsg=message)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[V1:.*]] = fir.embox %[[VAR_R]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<f32>) -> !fir.box<none>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V5]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(r,       result_image=1, stat=status, errmsg=message)

  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_I:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V5]], %[[V2]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(array_i)
   
  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_C:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1>>>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.ref<i32>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2x!fir.char<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_min_character(%[[V5]], %[[IMAGE_RESULT]], %[[V2]], %[[V3]], %[[V4]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(array_c, result_image=1) 
   
  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_D:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf64>>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xf64>>) -> !fir.box<none>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V2]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(array_d, result_image=1, stat=status)

  ! CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
  ! CHECK: %[[SHAPE_2:.*]] = fir.shape %[[C2_2:.*]] : (index) -> !fir.shape<1>
  ! CHECK: %[[V1:.*]] = fir.embox %[[ARRAY_C:.*]]#0(%[[SHAPE_2]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  ! CHECK: fir.store %[[C1_i32]] to %[[IMAGE_RESULT:.*]] : !fir.ref<i32>
  ! CHECK: %[[V2:.*]] = fir.embox %[[MESSAGE]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: %[[V3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! CHECK: %[[V4:.*]] = fir.convert %[[V1]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<none>
  ! CHECK: %[[V5:.*]] = fir.convert %[[V2]] : (!fir.box<!fir.char<1>>) -> !fir.box<!fir.char<1,?>>
  ! CHECK: fir.call @_QMprifPprif_co_min(%[[V4]], %[[IMAGE_RESULT]], %[[STATUS]], %[[V5]], %[[V3]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  call co_min(array_r, result_image=1, stat= status, errmsg=message)

end program

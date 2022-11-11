! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test that IO item list are lowered and passed correctly

! CHECK-LABEL: func @_QPpass_assumed_len_char_unformatted_io
subroutine pass_assumed_len_char_unformatted_io(c)
  character(*) :: c
  ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  write(1, rec=1) c
  ! CHECK: %[[box:.*]] = fir.embox %[[unbox]]#0 typeparams %[[unbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[box]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[castedBox]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
end

! CHECK-LABEL: func @_QPpass_assumed_len_char_array
subroutine pass_assumed_len_char_array(carray)
  character(*) :: carray(2, 3)
  ! CHECK-DAG: %[[unboxed:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG: %[[buffer:.*]] = fir.convert %[[unboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<2x3x!fir.char<1,?>>>
  ! CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
  ! CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %[[c2]], %[[c3]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) typeparams %[[unboxed]]#1 : (!fir.ref<!fir.array<2x3x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.box<!fir.array<2x3x!fir.char<1,?>>>
  ! CHECK: %[[descriptor:.*]] = fir.convert %[[box]] : (!fir.box<!fir.array<2x3x!fir.char<1,?>>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[descriptor]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  print *, carray
end

! CHECK-LABEL: func @_QPpass_array_slice_read(
! CHECK-SAME:            %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 5 : i32
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_4:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK:         %[[VAL_5:.*]] = fir.call @_FortranAioBeginExternalListInput(%[[VAL_1]], %[[VAL_3]], %[[VAL_4]]) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_6:.*]] = arith.constant 101 : i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 200 : i64
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:         %[[VAL_12:.*]] = fir.slice %[[VAL_7]], %[[VAL_11]], %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_13:.*]] = fir.rebox %[[VAL_0]] {{\[}}%[[VAL_12]]] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.array<50xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_15:.*]] = fir.call @_FortranAioInputDescriptor(%[[VAL_5]], %[[VAL_14]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_16:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_5]]) : (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }

subroutine pass_array_slice_read(x)
  real :: x(:)
  read(5, *) x(101:200:2)
end

! CHECK-LABEL: func @_QPpass_array_slice_write(
! CHECK-SAME:                   %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
! CHECK:         %[[VAL_4:.*]] = arith.constant {{[0-9]+}} : i32
! CHECK:         %[[VAL_5:.*]] = fir.call @_FortranAioBeginUnformattedOutput(%[[VAL_1]], %[[VAL_3]], %[[VAL_4]]) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> i64
! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranAioSetRec(%[[VAL_5]], %[[VAL_7]]) : (!fir.ref<i8>, i64) -> i1
! CHECK:         %[[VAL_9:.*]] = arith.constant 101 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 200 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = fir.slice %[[VAL_10]], %[[VAL_14]], %[[VAL_12]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_16:.*]] = fir.rebox %[[VAL_0]] {{\[}}%[[VAL_15]]] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.array<50xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_18:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_5]], %[[VAL_17]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_19:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_5]]) : (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }

subroutine pass_array_slice_write(x)
  real :: x(:)
  write(1, rec=1) x(101:200:2)
end


! CHECK-LABEL: func @_QPpass_vector_subscript_write(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<100xf32>>{{.*}}, %[[j:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}})
subroutine pass_vector_subscript_write(x, j)
  ! Check that a temp is made for array with vector subscript in output IO.
  integer :: j(10)
  real :: x(100)
  ! CHECK: %[[jload:.*]] = fir.array_load %[[j]](%{{.*}}) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
  ! CHECK: %[[xload:.*]] = fir.array_load %[[x]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<10xf32>
  ! CHECK: %[[tempload:.*]] = fir.array_load %[[temp]](%{{.*}}) : (!fir.heap<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[copy:.*]] = fir.do_loop
  ! CHECK:   %[[jfetch:.*]] = fir.array_fetch %[[jload]], %{{.*}} : (!fir.array<10xi32>, index) -> i32
  ! CHECK:   %[[jcast:.*]] = fir.convert %[[jfetch]] : (i32) -> index
  ! CHECK:   %[[jindex:.*]] = arith.subi %[[jcast]], %c1{{.*}} : index
  ! CHECK:   %[[xfetch:.*]] = fir.array_fetch %[[xload]], %[[jindex]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK:   %[[update:.*]] = fir.array_update %{{.*}}, %[[xfetch]], %{{.*}} : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK:   fir.result %[[update]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[tempload]], %[[copy]] to %[[temp]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.heap<!fir.array<10xf32>>
  ! CHECK: %[[embox:.*]] = fir.embox %[[temp]](%{{.*}}) : (!fir.heap<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  ! CHECK: %[[boxCast:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[boxCast]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<10xf32>>
  write(1, rec=1) x(j)
end

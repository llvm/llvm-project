!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine map_negative_bounds_allocatable_dtype()
    type derived_type
        real(4), pointer :: data(:,:,:) => null()
    end type
    type(derived_type), allocatable :: dtype(:,:)

    !$omp target map(tofrom: dtype(-1,1)%data)
        dtype(-1,1)%data(1,1,1) = 10
    !$omp end target
end subroutine

! CHECK:           %[[VAL_1:.*]] = arith.constant -1 : i64
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i64) -> index
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %{{.*}}, %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.type<_QFmap_negative_bounds_allocatable_dtypeTderived_type{data:!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>}>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_8:.*]]:3 = fir.box_dims %{{.*}}, %[[VAL_7]] : (!fir.box<!fir.heap<!fir.array<?x?x!fir.type<_QFmap_negative_bounds_allocatable_dtypeTderived_type{data:!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>}>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_9:.*]] = fir.shape_shift %[[VAL_6]]#0, %[[VAL_6]]#1, %[[VAL_8]]#0, %[[VAL_8]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:           %[[VAL_10:.*]] = fir.array_coor %{{.*}}(%[[VAL_9]]) %[[VAL_2]], %[[VAL_4]] : (!fir.heap<!fir.array<?x?x!fir.type<_QFmap_negative_bounds_allocatable_dtypeTderived_type{data:!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>}>>>, !fir.shapeshift<2>, index, index) -> !fir.ref<!fir.type<_QFmap_negative_bounds_allocatable_dtypeTderived_type{data:!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>}>>
! CHECK:           %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_10]], data : (!fir.ref<!fir.type<_QFmap_negative_bounds_allocatable_dtypeTderived_type{data:!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>
! CHECK:           %[[VAL_12:.*]] = fir.box_offset %[[VAL_11]] base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?x?x?xf32>>>
! CHECK:           %[[VAL_13:.*]] = omp.map.info var_ptr(%[[VAL_11]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>, f32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAL_12]] : !fir.llvm_ptr<!fir.ref<!fir.array<?x?x?xf32>>>) bounds({{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?x?x?xf32>>> {name = ""}
! CHECK:           %[[VAL_14:.*]] = omp.map.info var_ptr(%[[VAL_11]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>>, !fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x?xf32>>>> {name = {{.*}}}

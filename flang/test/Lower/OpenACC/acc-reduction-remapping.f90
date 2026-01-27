! Test remapping of variables appearing in OpenACC reduction clause
! to the related acc dialect data operation result.

! This tests checks how the hlfir.declare is recreated and used inside
! the acc compute region.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine scalar_combined(x, y)
  real :: y, x(100)
  !$acc parallel copyin(x) reduction(+:y)
  do i=1,100
    y = y + x(i)
  end do
  !$acc end parallel
end subroutine

subroutine scalar_split(x, y)
  real :: y, x(100)
  !$acc parallel copyin(x) copyout(y)
  !$acc loop  reduction(+:y)
  do i=1,100
    y = y + x(i)
  end do
  !$acc end parallel
end subroutine

subroutine array_combined(x, y, n)
  integer(8) :: n
  real :: y(n), x(100, n)
  !$acc parallel copyin(x) reduction(+:y)
  do j=1,n
    do i=1,100
      y(j) = y(j) + x(i, j)
    end do
  end do
  !$acc end parallel
end subroutine

subroutine array_split(x, y, n)
  integer(8) :: n
  real :: y(n), x(100, n)
  !$acc parallel copyin(x) copyout(y)
  !$acc loop reduction(+:y)
  do j=1,n
    do i=1,100
      y(j) = y(j) + x(i, j)
    end do
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL:   func.func @_QPscalar_combined(
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_Y:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFscalar_combinedEy"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[REDUCTION_Y:.*]] = acc.reduction varPtr(%[[DECLARE_Y]]#0 : !fir.ref<f32>) recipe(@reduction_add_ref_f32) -> !fir.ref<f32> {name = "y"}
! CHECK:           acc.parallel {{.*}} reduction(%[[REDUCTION_Y]] : !fir.ref<f32>) {
! CHECK:             %[[DUMMY_SCOPE_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[DECLARE_RED_PAR:.*]]:2 = hlfir.declare %[[REDUCTION_Y]] dummy_scope %[[DUMMY_SCOPE_1]] arg 2 {uniq_name = "_QFscalar_combinedEy"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             %[[PRIVATE_I:.*]] = acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK:             acc.loop private(%[[PRIVATE_I]] : !fir.ref<i32>) {{.*}} {
! CHECK:               %[[DECLARE_I:.*]]:2 = hlfir.declare %[[PRIVATE_I]] {uniq_name = "_QFscalar_combinedEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:               fir.store %[[VAL_0:.*]] to %[[DECLARE_I]]#0 : !fir.ref<i32>
! CHECK:               %[[LOAD_RED:.*]] = fir.load %[[DECLARE_RED_PAR]]#0 : !fir.ref<f32>
! CHECK:               {{.*}} = hlfir.designate {{.*}} : (!fir.ref<!fir.array<100xf32>>, i64) -> !fir.ref<f32>
! CHECK:               {{.*}} = fir.load {{.*}} : !fir.ref<f32>
! CHECK:               %[[ADDF:.*]] = arith.addf %[[LOAD_RED]], {{.*}} {{.*}}: f32
! CHECK:               hlfir.assign %[[ADDF]] to %[[DECLARE_RED_PAR]]#0 : f32, !fir.ref<f32>
! CHECK:               acc.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           return
! CHECK:         }
!
!
! CHECK-LABEL:   func.func @_QPscalar_split(
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_Y:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFscalar_splitEy"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[CREATE_Y:.*]] = acc.create varPtr(%[[DECLARE_Y]]#0 : !fir.ref<f32>) -> !fir.ref<f32> {dataClause = #acc<data_clause acc_copyout>, name = "y"}
! CHECK:           acc.parallel dataOperands({{.*}}%[[CREATE_Y]] : {{.*}}) {
! CHECK:             %[[DUMMY_SCOPE_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[DECLARE_Y_PAR:.*]]:2 = hlfir.declare %[[CREATE_Y]] dummy_scope %[[DUMMY_SCOPE_1]] arg 2 {uniq_name = "_QFscalar_splitEy"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             %[[REDUCTION_Y:.*]] = acc.reduction varPtr(%[[DECLARE_Y_PAR]]#0 : !fir.ref<f32>) recipe(@reduction_add_ref_f32) -> !fir.ref<f32> {name = "y"}
! CHECK:             %[[PRIVATE_I:.*]] = acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK:             acc.loop private(%[[PRIVATE_I]] : !fir.ref<i32>) reduction(%[[REDUCTION_Y]] : !fir.ref<f32>) {{.*}} {
! CHECK:               %[[DUMMY_SCOPE_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:               %[[DECLARE_RED:.*]]:2 = hlfir.declare %[[REDUCTION_Y]] dummy_scope %[[DUMMY_SCOPE_2]] arg 2 {uniq_name = "_QFscalar_splitEy"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:               %[[LOAD_RED:.*]] = fir.load %[[DECLARE_RED]]#0 : !fir.ref<f32>
! CHECK:               %[[ADDF:.*]] = arith.addf %[[LOAD_RED]], {{.*}} {{.*}}: f32
! CHECK:               hlfir.assign %[[ADDF]] to %[[DECLARE_RED]]#0 : f32, !fir.ref<f32>
! CHECK:               acc.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           return
! CHECK:         }


! CHECK-LABEL:   func.func @_QParray_combined(
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_N:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %[[DUMMY_SCOPE_0]] arg 3 {uniq_name = "_QFarray_combinedEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[DECLARE_Y:.*]]:2 = hlfir.declare %{{.*}}({{.*}}) dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFarray_combinedEy"} : (!fir.ref<!fir.array<?xf32>>, {{.*}}, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           %[[REDUCTION_Y:.*]] = acc.reduction var(%[[DECLARE_Y]]#0 : !fir.box<!fir.array<?xf32>>) recipe(@reduction_add_box_Uxf32) -> !fir.box<!fir.array<?xf32>> {name = "y"}
! CHECK:           acc.parallel dataOperands({{.*}}) reduction(%[[REDUCTION_Y]] : !fir.box<!fir.array<?xf32>>) {
! CHECK:             %[[DUMMY_SCOPE_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[BOX_ADDR_RED:.*]] = fir.box_addr %[[REDUCTION_Y]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:             %[[DECLARE_Y_PAR:.*]]:2 = hlfir.declare %[[BOX_ADDR_RED]]({{.*}}) dummy_scope %[[DUMMY_SCOPE_1]] arg 2 {uniq_name = "_QFarray_combinedEy"} : (!fir.ref<!fir.array<?xf32>>, {{.*}}, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:             %[[PRIVATE_J:.*]] = acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "j"}
! CHECK:             acc.loop private(%[[PRIVATE_J]] : !fir.ref<i32>) {{.*}} {
! CHECK:               %[[PRIVATE_I:.*]] = acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK:               acc.loop private(%[[PRIVATE_I]] : !fir.ref<i32>) {{.*}} {
! CHECK:                 %[[DESIGNATE_RED:.*]] = hlfir.designate %[[DECLARE_Y_PAR]]#0 ({{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:                 %[[LOAD_OLD:.*]] = fir.load %[[DESIGNATE_RED]] : !fir.ref<f32>
! CHECK:                 {{.*}} = hlfir.designate {{.*}} : (!fir.box<!fir.array<100x?xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:                 {{.*}} = fir.load {{.*}} : !fir.ref<f32>
! CHECK:                 %[[ADDF:.*]] = arith.addf %[[LOAD_OLD]], {{.*}} {{.*}}: f32
! CHECK:                 %[[DESIGNATE_RED2:.*]] = hlfir.designate %[[DECLARE_Y_PAR]]#0 ({{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:                 hlfir.assign %[[ADDF]] to %[[DESIGNATE_RED2]] : f32, !fir.ref<f32>
! CHECK:                 acc.yield
! CHECK:               }
! CHECK:               acc.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           return
! CHECK:         }


! CHECK-LABEL:   func.func @_QParray_split(
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_Y:.*]]:2 = hlfir.declare %{{.*}}({{.*}}) dummy_scope %[[DUMMY_SCOPE_0]] arg 2 {uniq_name = "_QFarray_splitEy"} : (!fir.ref<!fir.array<?xf32>>, {{.*}}, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           %[[CREATE_Y:.*]] = acc.create var(%[[DECLARE_Y]]#0 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "y"}
! CHECK:           acc.parallel dataOperands({{.*}}%[[CREATE_Y]] : {{.*}}) {
! CHECK:             %[[DUMMY_SCOPE_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:             %[[BOX_ADDR_Y:.*]] = fir.box_addr %[[CREATE_Y]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:             %[[DECLARE_Y_PAR:.*]]:2 = hlfir.declare %[[BOX_ADDR_Y]]({{.*}}) dummy_scope %[[DUMMY_SCOPE_1]] arg 2 {uniq_name = "_QFarray_splitEy"} : (!fir.ref<!fir.array<?xf32>>, {{.*}}, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:             %[[REDUCTION_Y:.*]] = acc.reduction var(%[[DECLARE_Y_PAR]]#0 : !fir.box<!fir.array<?xf32>>) recipe(@reduction_add_box_Uxf32) -> !fir.box<!fir.array<?xf32>> {name = "y"}
! CHECK:             %[[PRIVATE_J:.*]] = acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "j"}
! CHECK:             acc.loop private(%[[PRIVATE_J]] : !fir.ref<i32>) reduction(%[[REDUCTION_Y]] : !fir.box<!fir.array<?xf32>>) {{.*}} {
! CHECK:               %[[BOX_ADDR_RED:.*]] = fir.box_addr %[[REDUCTION_Y]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:               %[[DUMMY_SCOPE_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:               %[[DECLARE_Y_LOOP_PAR:.*]]:2 = hlfir.declare %[[BOX_ADDR_RED]]({{.*}}) dummy_scope %[[DUMMY_SCOPE_2]] arg 2 {uniq_name = "_QFarray_splitEy"} : (!fir.ref<!fir.array<?xf32>>, {{.*}}, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:               %[[PRIVATE_I:.*]] = acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK:               acc.loop private(%[[PRIVATE_I]] : !fir.ref<i32>) {{.*}} {
! CHECK:                 %[[DESIGNATE_RED:.*]] = hlfir.designate %[[DECLARE_Y_LOOP_PAR]]#0 ({{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:                 %[[LOAD_OLD:.*]] = fir.load %[[DESIGNATE_RED]] : !fir.ref<f32>
! CHECK:                 {{.*}} = hlfir.designate {{.*}} : (!fir.box<!fir.array<100x?xf32>>, i64, i64) -> !fir.ref<f32>
! CHECK:                 {{.*}} = fir.load {{.*}} : !fir.ref<f32>
! CHECK:                 %[[ADDF:.*]] = arith.addf %[[LOAD_OLD]], {{.*}} {{.*}}: f32
! CHECK:                 %[[DESIGNATE_RED2:.*]] = hlfir.designate %[[DECLARE_Y_LOOP_PAR]]#0 ({{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK:                 hlfir.assign %[[ADDF]] to %[[DESIGNATE_RED2]] : f32, !fir.ref<f32>
! CHECK:                 acc.yield
! CHECK:               }
! CHECK:               acc.yield
! CHECK:             }
! CHECK:             acc.yield
! CHECK:           }
! CHECK:           return
! CHECK:         }

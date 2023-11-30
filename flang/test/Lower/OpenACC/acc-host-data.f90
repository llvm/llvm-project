! This test checks lowering of OpenACC host_data directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine acc_host_data()
  real, dimension(10) :: a
  logical :: ifCondition = .TRUE.

! CHECK: %[[A:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "a", uniq_name = "_QFacc_host_dataEa"}
! CHECK: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[IFCOND:.*]] = fir.address_of(@_QFacc_host_dataEifcondition) : !fir.ref<!fir.logical<4>>
! CHECK: %[[DECLIFCOND:.*]]:2 = hlfir.declare %[[IFCOND]]

  !$acc host_data use_device(a)
  !$acc end host_data

! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[DA:.*]] = acc.use_device varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK: acc.host_data dataOperands(%[[DA]] : !fir.ref<!fir.array<10xf32>>)

  !$acc host_data use_device(a) if_present
  !$acc end host_data

! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[DA:.*]] = acc.use_device varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK: acc.host_data dataOperands(%[[DA]] : !fir.ref<!fir.array<10xf32>>) {
! CHECK: } attributes {ifPresent}

  !$acc host_data use_device(a) if(ifCondition)
  !$acc end host_data

! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[DA:.*]] = acc.use_device varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK: %[[LOAD_IFCOND:.*]] = fir.load %[[DECLIFCOND]]#0 : !fir.ref<!fir.logical<4>>
! CHECK: %[[IFCOND_I1:.*]] = fir.convert %[[LOAD_IFCOND]] : (!fir.logical<4>) -> i1
! CHECK: acc.host_data if(%[[IFCOND_I1]]) dataOperands(%[[DA]] : !fir.ref<!fir.array<10xf32>>)

  !$acc host_data use_device(a) if(.true.)
  !$acc end host_data

! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[DA:.*]] = acc.use_device varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK: acc.host_data dataOperands(%[[DA]] : !fir.ref<!fir.array<10xf32>>)

  !$acc host_data use_device(a) if(.false.)
    a = 1.0
  !$acc end host_data

! CHECK-NOT: acc.host_data
! CHECK: hlfir.assign %{{.*}} to %[[DECLA]]#0

end subroutine

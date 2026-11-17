! Test a COMMON block declared in a module and named in a data clause of a
! module procedure that does not otherwise reference any of its members.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_module_common
  integer :: a, b
  common /c/ a, b
contains
  subroutine enter_data_create()
    !$acc enter data create(/c/)
  end subroutine
  subroutine exit_data_copyout()
    !$acc exit data copyout(/c/)
  end subroutine
  subroutine update_device()
    !$acc update device(/c/)
  end subroutine
  subroutine data_present()
    !$acc data present(/c/)
    !$acc end data
  end subroutine
end module

! On a compute construct every member of the block is rebound in the region,
! including the members that the region does not reference.
module acc_module_common_compute
  integer :: a, b
  common /cc/ a, b
contains
  subroutine parallel_copyin_one_member(ierr)
    integer :: ierr
    !$acc parallel copyin(/cc/) copy(ierr)
    if (b /= 1) ierr = 1
    !$acc end parallel
  end subroutine
end module

module acc_module_common_link
  integer :: a, b
  common /cl/ a, b
  !$acc declare link(/cl/)
contains
  subroutine enter_data_copyin()
    !$acc enter data copyin(/cl/)
  end subroutine
end module

! EQUIVALENCE extends the block with an object that no COMMON statement names.
module acc_module_common_equivalence
  integer :: a, b
  integer :: e(4)
  common /ce/ a, b
  equivalence (b, e(1))
contains
  subroutine parallel_copy()
    !$acc parallel copy(/ce/)
    !$acc end parallel
  end subroutine
  subroutine serial_firstprivate_one_member(ierr)
    integer :: ierr
    !$acc serial firstprivate(/ce/) copy(ierr)
    if (b /= 1) ierr = 1
    !$acc end serial
  end subroutine
end module

! CHECK-LABEL: func.func @_QMacc_module_commonPenter_data_create()
! CHECK:         %[[C:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[CREATE:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("c")
! CHECK:         acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<8xi8>>)

! CHECK-LABEL: func.func @_QMacc_module_commonPexit_data_copyout()
! CHECK:         %[[C:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) dataClause(acc_copyout) structured(false) name("c")
! CHECK:         acc.exit_data dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<8xi8>>)
! CHECK:         acc.copyout accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<8xi8>>) to varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("c")

! CHECK-LABEL: func.func @_QMacc_module_commonPupdate_device()
! CHECK:         %[[C:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[UPD:.*]] = acc.update_device varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("c")
! CHECK:         acc.update dataOperands(%[[UPD]] : !fir.ref<!fir.array<8xi8>>)

! CHECK-LABEL: func.func @_QMacc_module_commonPdata_present()
! CHECK:         %[[C:.*]] = fir.address_of(@c_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[PRESENT:.*]] = acc.present varPtr(%[[C]] : !fir.ref<!fir.array<8xi8>>) name("c")
! CHECK:         acc.data dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<8xi8>>)

! CHECK-LABEL: func.func @_QMacc_module_common_computePparallel_copyin_one_member(
! CHECK:         %[[CC:.*]] = fir.address_of(@cc_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[COPYIN:.*]] = acc.copyin varPtr(%[[CC]] : !fir.ref<!fir.array<8xi8>>) name("cc")
! CHECK:         acc.parallel dataOperands(%[[COPYIN]], %{{.*}} : !fir.ref<!fir.array<8xi8>>, !fir.ref<i32>) {
! CHECK:           hlfir.declare %{{.*}} storage(%[[COPYIN]][0]) uniq_name("_QMacc_module_common_computeEa")
! CHECK:           %[[B:.*]]:2 = hlfir.declare %{{.*}} storage(%[[COPYIN]][4]) uniq_name("_QMacc_module_common_computeEb")
! CHECK:           fir.load %[[B]]#0 : !fir.ref<i32>

! CHECK-LABEL: func.func @_QMacc_module_common_linkPenter_data_copyin()
! CHECK:         %[[CL:.*]] = fir.address_of(@cl_) : !fir.ref<!fir.array<8xi8>>
! CHECK:         %[[COPYIN:.*]] = acc.copyin varPtr(%[[CL]] : !fir.ref<!fir.array<8xi8>>) structured(false) name("cl")
! CHECK:         acc.enter_data dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<8xi8>>)

! CHECK-LABEL: func.func @_QMacc_module_common_equivalencePparallel_copy()
! CHECK:         %[[CE:.*]] = fir.address_of(@ce_) : !fir.ref<!fir.array<20xi8>>
! CHECK:         %[[COPY:.*]] = acc.copyin varPtr(%[[CE]] : !fir.ref<!fir.array<20xi8>>) dataClause(acc_copy) name("ce")
! CHECK:         acc.parallel dataOperands(%[[COPY]] : !fir.ref<!fir.array<20xi8>>) {
! CHECK:           hlfir.declare %{{.*}} storage(%[[COPY]][0]) uniq_name("_QMacc_module_common_equivalenceEa")
! CHECK:           hlfir.declare %{{.*}} storage(%[[COPY]][4]) uniq_name("_QMacc_module_common_equivalenceEb")
! CHECK:           hlfir.declare %{{.*}}(%{{.*}}) storage(%[[COPY]][4]) uniq_name("_QMacc_module_common_equivalenceEe")

! CHECK-LABEL: func.func @_QMacc_module_common_equivalencePserial_firstprivate_one_member(
! CHECK:         %[[CE:.*]] = fir.address_of(@ce_) : !fir.ref<!fir.array<20xi8>>
! CHECK:         %[[FP:.*]] = acc.firstprivate varPtr(%[[CE]] : !fir.ref<!fir.array<20xi8>>) recipe(@firstprivatization_optional_ref_20xi8) name("ce")
! CHECK:         acc.serial dataOperands(%{{.*}} : !fir.ref<i32>) firstprivate(%[[FP]] : !fir.ref<!fir.array<20xi8>>) {
! CHECK:           hlfir.declare %{{.*}} storage(%[[FP]][0]) uniq_name("_QMacc_module_common_equivalenceEa")
! CHECK:           %[[B:.*]]:2 = hlfir.declare %{{.*}} storage(%[[FP]][4]) uniq_name("_QMacc_module_common_equivalenceEb")
! CHECK:           hlfir.declare %{{.*}}(%{{.*}}) storage(%[[FP]][4]) uniq_name("_QMacc_module_common_equivalenceEe")
! CHECK:           fir.load %[[B]]#0 : !fir.ptr<i32>

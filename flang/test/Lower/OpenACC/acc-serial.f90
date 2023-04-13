! This test checks lowering of OpenACC serial directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_serial
  integer :: i, j

  integer :: async = 1
  integer :: wait1 = 1
  integer :: wait2 = 2
  integer :: numGangs = 1
  integer :: numWorkers = 10
  integer :: vectorLength = 128
  logical :: ifCondition = .TRUE.
  real, dimension(10, 10) :: a, b, c
  real, pointer :: d, e

! CHECK: [[A:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
! CHECK: [[B:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
! CHECK: [[C:%.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
! CHECK: [[D:%.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}
! CHECK: [[E:%.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "e", uniq_name = "{{.*}}Ee"}
! CHECK: [[IFCONDITION:%.*]] = fir.address_of(@{{.*}}ifcondition) : !fir.ref<!fir.logical<4>>

  !$acc serial
  !$acc end serial

! CHECK:      acc.serial {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial async
  !$acc end serial

! CHECK:      acc.serial {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {asyncAttr}

  !$acc serial async(1)
  !$acc end serial

! CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.serial async([[ASYNC1]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial async(async)
  !$acc end serial

! CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.serial async([[ASYNC2]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial wait
  !$acc end serial

! CHECK:      acc.serial {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {waitAttr}

  !$acc serial wait(1)
  !$acc end serial

! CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.serial wait([[WAIT1]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial wait(1, 2)
  !$acc end serial

! CHECK:      [[WAIT2:%.*]] = arith.constant 1 : i32
! CHECK:      [[WAIT3:%.*]] = arith.constant 2 : i32
! CHECK:      acc.serial wait([[WAIT2]], [[WAIT3]] : i32, i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial wait(wait1, wait2)
  !$acc end serial

! CHECK:      [[WAIT4:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      [[WAIT5:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.serial wait([[WAIT4]], [[WAIT5]] : i32, i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial if(.TRUE.)
  !$acc end serial

! CHECK:      [[IF1:%.*]] = arith.constant true
! CHECK:      acc.serial if([[IF1]]) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial if(ifCondition)
  !$acc end serial

! CHECK:      [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:      [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK:      acc.serial if([[IF2]]) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial self(.TRUE.)
  !$acc end serial

! CHECK:      [[SELF1:%.*]] = arith.constant true
! CHECK:      acc.serial self([[SELF1]]) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial self
  !$acc end serial

! CHECK:      acc.serial {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {selfAttr}

  !$acc serial self(ifCondition)
  !$acc end serial

! CHECK:      [[SELF2:%.*]] = fir.convert [[IFCONDITION]] : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:      acc.serial self([[SELF2]]) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial copy(a, b, c)
  !$acc end serial

! CHECK:      acc.serial copy([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial copy(a) copy(b) copy(c)
  !$acc end serial

! CHECK:      acc.serial copy([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial copyin(a) copyin(readonly: b, c)
  !$acc end serial

! CHECK:      acc.serial copyin([[A]] : !fir.ref<!fir.array<10x10xf32>>) copyin_readonly([[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial copyout(a) copyout(zero: b) copyout(c)
  !$acc end serial

! CHECK:      acc.serial copyout([[A]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) copyout_zero([[B]] : !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial create(a, b) create(zero: c)
  !$acc end serial

! CHECK:      acc.serial create([[A]], [[B]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) create_zero([[C]] : !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial no_create(a, b) create(zero: c)
  !$acc end serial

! CHECK:      acc.serial create_zero([[C]] : !fir.ref<!fir.array<10x10xf32>>) no_create([[A]], [[B]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial present(a, b, c)
  !$acc end serial

! CHECK:      acc.serial present([[A]], [[B]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial deviceptr(a) deviceptr(c)
  !$acc end serial

! CHECK:      acc.serial deviceptr([[A]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial attach(d, e)
  !$acc end serial

! CHECK:      acc.serial attach([[D]], [[E]] : !fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial private(a) firstprivate(b) private(c)
  !$acc end serial

! CHECK:      acc.serial firstprivate([[B]] : !fir.ref<!fir.array<10x10xf32>>) private([[A]], [[C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

end subroutine

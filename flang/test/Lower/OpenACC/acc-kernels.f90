! This test checks lowering of OpenACC kernels construct.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_kernels
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

! CHECK: %[[A:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
! CHECK: %[[B:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
! CHECK: %[[C:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
! CHECK: %[[D:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}
! CHECK: %[[E:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "e", uniq_name = "{{.*}}Ee"}
! CHECK: %[[IFCONDITION:.*]] = fir.address_of(@{{.*}}ifcondition) : !fir.ref<!fir.logical<4>>

  !$acc kernels
  !$acc end kernels

! CHECK:      acc.kernels  {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels async
  !$acc end kernels

! CHECK:      acc.kernels  {
! CHECK:        acc.terminator
! CHECK-NEXT: } attributes {asyncAttr}

  !$acc kernels async(1)
  !$acc end kernels

! CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels  async([[ASYNC1]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels async(async)
  !$acc end kernels

! CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels  async([[ASYNC2]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels wait
  !$acc end kernels

! CHECK:      acc.kernels  {
! CHECK:        acc.terminator
! CHECK-NEXT: } attributes {waitAttr}

  !$acc kernels wait(1)
  !$acc end kernels

! CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels  wait([[WAIT1]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels wait(1, 2)
  !$acc end kernels

! CHECK:      [[WAIT2:%.*]] = arith.constant 1 : i32
! CHECK:      [[WAIT3:%.*]] = arith.constant 2 : i32
! CHECK:      acc.kernels  wait([[WAIT2]], [[WAIT3]] : i32, i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels wait(wait1, wait2)
  !$acc end kernels

! CHECK:      [[WAIT4:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      [[WAIT5:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels  wait([[WAIT4]], [[WAIT5]] : i32, i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels num_gangs(1)
  !$acc end kernels

! CHECK:      [[NUMGANGS1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels  num_gangs([[NUMGANGS1]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels num_gangs(numGangs)
  !$acc end kernels

! CHECK:      [[NUMGANGS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels  num_gangs([[NUMGANGS2]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels num_workers(10)
  !$acc end kernels

! CHECK:      [[NUMWORKERS1:%.*]] = arith.constant 10 : i32
! CHECK:      acc.kernels  num_workers([[NUMWORKERS1]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels num_workers(numWorkers)
  !$acc end kernels

! CHECK:      [[NUMWORKERS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels  num_workers([[NUMWORKERS2]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels vector_length(128)
  !$acc end kernels

! CHECK:      [[VECTORLENGTH1:%.*]] = arith.constant 128 : i32
! CHECK:      acc.kernels  vector_length([[VECTORLENGTH1]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels vector_length(vectorLength)
  !$acc end kernels

! CHECK:      [[VECTORLENGTH2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels  vector_length([[VECTORLENGTH2]] : i32) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels if(.TRUE.)
  !$acc end kernels

! CHECK:      [[IF1:%.*]] = arith.constant true
! CHECK:      acc.kernels  if([[IF1]]) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels if(ifCondition)
  !$acc end kernels

! CHECK:      [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:      [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK:      acc.kernels  if([[IF2]]) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels self(.TRUE.)
  !$acc end kernels

! CHECK:      [[SELF1:%.*]] = arith.constant true
! CHECK:      acc.kernels  self([[SELF1]]) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels self
  !$acc end kernels

! CHECK:      acc.kernels  {
! CHECK:        acc.terminator
! CHECK-NEXT: } attributes {selfAttr}

  !$acc kernels self(ifCondition)
  !$acc end kernels

! CHECK:      %[[SELF2:.*]] = fir.convert %[[IFCONDITION]] : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:      acc.kernels self(%[[SELF2]]) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels copy(a, b, c)
  !$acc end kernels

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 3 : i64, name = "a"} 
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 3 : i64, name = "b"} 
! CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 3 : i64, name = "c"} 
! CHECK:      acc.kernels  dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = 3 : i64, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = 3 : i64, name = "b"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = 3 : i64, name = "c"}

  !$acc kernels copy(a) copy(b) copy(c)
  !$acc end kernels

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 3 : i64, name = "a"} 
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 3 : i64, name = "b"} 
! CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 3 : i64, name = "c"} 
! CHECK:      acc.kernels dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = 3 : i64, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = 3 : i64, name = "b"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = 3 : i64, name = "c"}

  !$acc kernels copyin(a) copyin(readonly: b, c)
  !$acc end kernels

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 2 : i64, name = "b"} 
! CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 2 : i64, name = "c"} 
! CHECK:      acc.kernels dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels copyout(a) copyout(zero: b) copyout(c)
  !$acc end kernels

! CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 4 : i64, name = "a"}
! CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 4 : i64, name = "b"}
! CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 4 : i64, name = "c"}
! CHECK:      acc.kernels dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK: acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a"}
! CHECK: acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) {name = "b"}
! CHECK: acc.copyout accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) {name = "c"}

  !$acc kernels create(a, b) create(zero: c)
  !$acc end kernels

! CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 8 : i64, name = "c"}
! CHECK:      acc.kernels dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK: acc.delete accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = 7 : i64, name = "a"}
! CHECK: acc.delete accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = 7 : i64, name = "b"}
! CHECK: acc.delete accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = 8 : i64, name = "c"}

  !$acc kernels no_create(a, b) create(zero: c)
  !$acc end kernels

! CHECK: %[[NO_CREATE_A:.*]] = acc.nocreate varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK: %[[NO_CREATE_B:.*]] = acc.nocreate varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 8 : i64, name = "c"}
! CHECK:      acc.kernels dataOperands(%[[NO_CREATE_A]], %[[NO_CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels present(a, b, c)
  !$acc end kernels

! CHECK: %[[PRESENT_A:.*]] = acc.present varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK: %[[PRESENT_B:.*]] = acc.present varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! CHECK: %[[PRESENT_C:.*]] = acc.present varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! CHECK:      acc.kernels dataOperands(%[[PRESENT_A]], %[[PRESENT_B]], %[[PRESENT_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels deviceptr(a) deviceptr(c)
  !$acc end kernels

! CHECK:      %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK:      %[[DEVICEPTR_C:.*]] = acc.deviceptr varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! CHECK:      acc.kernels dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels attach(d, e)
  !$acc end kernels

! CHECK:      %[[BOX_D:.*]] = fir.load %[[D]] : !fir.ref<!fir.box<!fir.ptr<f32>>> 
! CHECK:      %[[BOX_ADDR_D:.*]] = fir.box_addr %[[BOX_D]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32> 
! CHECK:      %[[ATTACH_D:.*]] = acc.attach varPtr(%[[BOX_ADDR_D]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "d"} 
! CHECK:      %[[BOX_E:.*]] = fir.load %[[E]] : !fir.ref<!fir.box<!fir.ptr<f32>>> 
! CHECK:      %[[BOX_ADDR_E:.*]] = fir.box_addr %[[BOX_E]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32> 
! CHECK:      %[[ATTACH_E:.*]] = acc.attach varPtr(%[[BOX_ADDR_E]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "e"} 
! CHECK:      acc.kernels dataOperands(%[[ATTACH_D]], %[[ATTACH_E]] : !fir.ptr<f32>, !fir.ptr<f32>) { 
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

end subroutine acc_kernels

! This test checks lowering of OpenACC parallel directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_section_ext10xext10_ref_10x10xf32 : !fir.ref<!fir.array<10x10xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<10x10xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<10x10xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<10x10xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<10x10xf32>>
! CHECK: } copy {
! CHECK: ^bb0(%arg0: !fir.ref<!fir.array<10x10xf32>>, %arg1: !fir.ref<!fir.array<10x10xf32>>):
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_10x10xf32 : !fir.ref<!fir.array<10x10xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<10x10xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<10x10xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<10x10xf32>>
! CHECK: }

! CHECK-LABEL: func.func @_QPacc_parallel()

subroutine acc_parallel
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
  integer :: reduction_i
  real :: reduction_r

! CHECK: %[[A:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
! CHECK: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[B:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
! CHECK: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
! CHECK: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]
! CHECK: %[[D:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}
! CHECK: %[[DECLD:.*]]:2 = hlfir.declare %[[D]]
! CHECK: %[[E:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "e", uniq_name = "{{.*}}Ee"}
! CHECK: %[[DECLE:.*]]:2 = hlfir.declare %[[E]]
! CHECK: %[[IFCONDITION:.*]] = fir.address_of(@{{.*}}ifcondition) : !fir.ref<!fir.logical<4>>
! CHECK: %[[DECLIFCONDITION:.*]]:2 = hlfir.declare %[[IFCONDITION]]

  !$acc parallel
  !$acc end parallel

! CHECK:      acc.parallel {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel async
  !$acc end parallel

! CHECK: acc.parallel {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {asyncOnly = [#acc.device_type<none>]}

  !$acc parallel async(1)
  !$acc end parallel

! CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.parallel async([[ASYNC1]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel async(async)
  !$acc end parallel

! CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-NEXT:  acc.parallel async([[ASYNC2]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel wait
  !$acc end parallel

! CHECK:      acc.parallel wait {
! CHECK:        acc.yield
! CHECK-NEXT: }

  !$acc parallel wait(1)
  !$acc end parallel

! CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.parallel wait({[[WAIT1]] : i32}) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel wait(1, 2)
  !$acc end parallel

! CHECK:      [[WAIT2:%.*]] = arith.constant 1 : i32
! CHECK:      [[WAIT3:%.*]] = arith.constant 2 : i32
! CHECK:      acc.parallel wait({[[WAIT2]] : i32, [[WAIT3]] : i32}) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel wait(wait1, wait2)
  !$acc end parallel

! CHECK:      [[WAIT4:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      [[WAIT5:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel wait({[[WAIT4]] : i32, [[WAIT5]] : i32}) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel num_gangs(1)
  !$acc end parallel

! CHECK:      [[NUMGANGS1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.parallel num_gangs({[[NUMGANGS1]] : i32}) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel num_gangs(numGangs)
  !$acc end parallel

! CHECK:      [[NUMGANGS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel num_gangs({[[NUMGANGS2]] : i32}) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel num_gangs(1, 1, 1)
  !$acc end parallel

! CHECK:      acc.parallel num_gangs({%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32}) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel num_workers(10)
  !$acc end parallel

! CHECK:      [[NUMWORKERS1:%.*]] = arith.constant 10 : i32
! CHECK:      acc.parallel num_workers([[NUMWORKERS1]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel num_workers(numWorkers)
  !$acc end parallel

! CHECK:      [[NUMWORKERS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel num_workers([[NUMWORKERS2]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel vector_length(128)
  !$acc end parallel

! CHECK:      [[VECTORLENGTH1:%.*]] = arith.constant 128 : i32
! CHECK:      acc.parallel vector_length([[VECTORLENGTH1]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel vector_length(vectorLength)
  !$acc end parallel

! CHECK:      [[VECTORLENGTH2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel vector_length([[VECTORLENGTH2]] : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel if(.TRUE.)
  !$acc end parallel

! CHECK:      [[IF1:%.*]] = arith.constant true
! CHECK:      acc.parallel if([[IF1]]) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel if(ifCondition)
  !$acc end parallel

! CHECK:      [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:      [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK:      acc.parallel if([[IF2]]) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel self(.TRUE.)
  !$acc end parallel

! CHECK:      [[SELF1:%.*]] = arith.constant true
! CHECK:      acc.parallel self([[SELF1]]) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel self
  !$acc end parallel

! CHECK:      acc.parallel {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {selfAttr}

  !$acc parallel self(ifCondition)
  !$acc end parallel

! CHECK:      %[[SELF2:.*]] = fir.convert %[[DECLIFCONDITION]]#1 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:      acc.parallel self(%[[SELF2]]) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel copy(a, b, c)
  !$acc end parallel

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "c"}
! CHECK:      acc.parallel dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "c"}

  !$acc parallel copy(a) copy(b) copy(c)
  !$acc end parallel


! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "c"}
! CHECK:      acc.parallel dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "c"}

  !$acc parallel copyin(a) copyin(readonly: b, c)
  !$acc end parallel

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! CHECK:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "c"}
! CHECK:      acc.parallel dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_copyin>, name = "a"}
! CHECK:      acc.delete accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! CHECK:      acc.delete accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_copyin_readonly>, name = "c"}

  !$acc parallel copyout(a) copyout(zero: b) copyout(c)
  !$acc end parallel

! CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "c"}
! CHECK:      acc.parallel dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK: acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a"}
! CHECK: acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "b"}
! CHECK: acc.copyout accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "c"}

  !$acc parallel create(a, b) create(zero: c)
  !$acc end parallel

! CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "c"}
! CHECK:      acc.parallel dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK: acc.delete accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_create>, name = "a"}
! CHECK: acc.delete accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_create>, name = "b"}
! CHECK: acc.delete accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_create_zero>, name = "c"}

  !$acc parallel create(c) copy(b) create(a)
  !$acc end parallel
! CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! CHECK: %[[COPY_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK: acc.parallel   dataOperands(%[[CREATE_C]], %[[COPY_B]], %[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {

  !$acc parallel no_create(a, b) create(zero: c)
  !$acc end parallel

! CHECK: %[[NO_CREATE_A:.*]] = acc.nocreate varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK: %[[NO_CREATE_B:.*]] = acc.nocreate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "c"}
! CHECK:      acc.parallel dataOperands(%[[NO_CREATE_A]], %[[NO_CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK: acc.delete accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_create_zero>, name = "c"}


  !$acc parallel present(a, b, c)
  !$acc end parallel

! CHECK: %[[PRESENT_A:.*]] = acc.present varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK: %[[PRESENT_B:.*]] = acc.present varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! CHECK: %[[PRESENT_C:.*]] = acc.present varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! CHECK:      acc.parallel dataOperands(%[[PRESENT_A]], %[[PRESENT_B]], %[[PRESENT_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel deviceptr(a) deviceptr(c)
  !$acc end parallel

! CHECK: %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK: %[[DEVICEPTR_C:.*]] = acc.deviceptr varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! CHECK:      acc.parallel dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel attach(d, e)
  !$acc end parallel

! CHECK: %[[BOX_D:.*]] = fir.load %[[DECLD]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[BOX_ADDR_D:.*]] = fir.box_addr %[[BOX_D]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK: %[[ATTACH_D:.*]] = acc.attach varPtr(%[[BOX_ADDR_D]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "d"}
! CHECK: %[[BOX_E:.*]] = fir.load %[[DECLE]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[BOX_ADDR_E:.*]] = fir.box_addr %[[BOX_E]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK: %[[ATTACH_E:.*]] = acc.attach varPtr(%[[BOX_ADDR_E]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "e"}
! CHECK: acc.parallel dataOperands(%[[ATTACH_D]], %[[ATTACH_E]] : !fir.ptr<f32>, !fir.ptr<f32>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK: acc.detach accPtr(%[[ATTACH_D]] : !fir.ptr<f32>) {dataClause = #acc<data_clause acc_attach>, name = "d"}
! CHECK: acc.detach accPtr(%[[ATTACH_E]] : !fir.ptr<f32>) {dataClause = #acc<data_clause acc_attach>, name = "e"}

!$acc parallel private(a) firstprivate(b) private(c) async(1)
!$acc end parallel

! CHECK:      %[[ACC_PRIVATE_A:.*]] = acc.private varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) async([[ASYNC3:%.*]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! CHECK:      %[[ACC_FPRIVATE_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) async([[ASYNC3]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! CHECK:      %[[ACC_PRIVATE_C:.*]] = acc.private varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) async([[ASYNC3]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! CHECK:      acc.parallel async([[ASYNC3]]) firstprivate(@firstprivatization_section_ext10xext10_ref_10x10xf32 -> %[[ACC_FPRIVATE_B]] : !fir.ref<!fir.array<10x10xf32>>) private(@privatization_ref_10x10xf32 -> %[[ACC_PRIVATE_A]] : !fir.ref<!fir.array<10x10xf32>>, @privatization_ref_10x10xf32 -> %[[ACC_PRIVATE_C]] : !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

!$acc parallel reduction(+:reduction_r) reduction(*:reduction_i)
!$acc end parallel

! CHECK:      acc.parallel reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

!$acc parallel default(none)
!$acc end parallel

! CHECK: acc.parallel {
! CHECK: } attributes {defaultAttr = #acc<defaultvalue none>}

!$acc parallel default(present)
!$acc end parallel

! CHECK: acc.parallel {
! CHECK: } attributes {defaultAttr = #acc<defaultvalue present>}

end subroutine acc_parallel

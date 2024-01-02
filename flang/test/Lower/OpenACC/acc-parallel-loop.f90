! This test checks lowering of OpenACC parallel loop combined directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPacc_parallel_loop()

subroutine acc_parallel_loop
  integer :: i, j

  integer :: async = 1
  integer :: wait1 = 1
  integer :: wait2 = 2
  integer :: numGangs = 1
  integer :: numWorkers = 10
  integer :: vectorLength = 128
  logical :: ifCondition = .TRUE.
  integer, parameter :: n = 10
  real, dimension(n) :: a, b, c
  real, dimension(n, n) :: d, e
  real, pointer :: f, g
  integer :: reduction_i
  real :: reduction_r

  integer :: gangNum = 8
  integer :: gangStatic = 8
  integer :: vectorNum = 128
  integer, parameter :: tileSize = 2

! CHECK: %[[A:.*]] = fir.alloca !fir.array<10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
! CHECK: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[B:.*]] = fir.alloca !fir.array<10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
! CHECK: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca !fir.array<10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
! CHECK: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]
! CHECK: %[[F:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "f", uniq_name = "{{.*}}Ef"}
! CHECK: %[[DECLF:.*]]:2 = hlfir.declare %[[F]]
! CHECK: %[[G:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "g", uniq_name = "{{.*}}Eg"}
! CHECK: %[[DECLG:.*]]:2 = hlfir.declare %[[G]]
! CHECK: %[[IFCONDITION:.*]] = fir.address_of(@{{.*}}ifcondition) : !fir.ref<!fir.logical<4>>
! CHECK: %[[DECLIFCONDITION:.*]]:2 = hlfir.declare %[[IFCONDITION]]

  !$acc parallel loop
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop async
  DO i = 1, n
    a(i) = b(i)
  END DO
  !$acc end parallel loop

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {asyncOnly = [#acc.device_type<none>]}

  !$acc parallel loop async(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.parallel async([[ASYNC1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop async(async)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel async([[ASYNC2]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop wait
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {waitOnly = [#acc.device_type<none>]}

  !$acc parallel loop wait(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.parallel wait({[[WAIT1]] : i32}) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop wait(1, 2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT2:%.*]] = arith.constant 1 : i32
! CHECK:      [[WAIT3:%.*]] = arith.constant 2 : i32
! CHECK:      acc.parallel wait({[[WAIT2]] : i32, [[WAIT3]] : i32}) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop wait(wait1, wait2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT4:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      [[WAIT5:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel wait({[[WAIT4]] : i32, [[WAIT5]] : i32}) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop num_gangs(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMGANGS1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.parallel num_gangs({[[NUMGANGS1]] : i32}) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop num_gangs(numGangs)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMGANGS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel num_gangs({[[NUMGANGS2]] : i32}) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop num_workers(10)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMWORKERS1:%.*]] = arith.constant 10 : i32
! CHECK:      acc.parallel num_workers([[NUMWORKERS1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop num_workers(numWorkers)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMWORKERS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel num_workers([[NUMWORKERS2]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop vector_length(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[VECTORLENGTH1:%.*]] = arith.constant 128 : i32
! CHECK:      acc.parallel vector_length([[VECTORLENGTH1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop vector_length(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[VECTORLENGTH2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.parallel vector_length([[VECTORLENGTH2]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop if(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[IF1:%.*]] = arith.constant true
! CHECK:      acc.parallel if([[IF1]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop if(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:      [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK:      acc.parallel if([[IF2]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop self(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[SELF1:%.*]] = arith.constant true
! CHECK:      acc.parallel self([[SELF1]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop self
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {selfAttr}

  !$acc parallel loop self(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[SELF2:.*]] = fir.convert %[[DECLIFCONDITION]]#1 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:      acc.parallel self(%[[SELF2]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop copy(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.parallel dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc parallel loop copy(a) copy(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.parallel dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc parallel loop copyin(a) copyin(readonly: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! CHECK:      acc.parallel dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop copyout(a) copyout(zero: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! CHECK:      acc.parallel dataOperands(%[[CREATE_A]], %[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {name = "a"}
! CHECK:      acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {name = "b"}

  !$acc parallel loop create(b) create(zero: a)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "a"}
! CHECK:      acc.parallel dataOperands(%[[CREATE_B]], %[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_create>, name = "b"}
! CHECK:      acc.delete accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_create_zero>, name = "a"}

  !$acc parallel loop no_create(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.parallel dataOperands(%[[NOCREATE_A]], %[[NOCREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop present(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[PRESENT_B:.*]] = acc.present varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.parallel dataOperands(%[[PRESENT_A]], %[[PRESENT_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop deviceptr(a) deviceptr(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.parallel dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop attach(f, g)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[BOX_F:.*]] = fir.load %[[DECLF]]#1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:      %[[BOX_ADDR_F:.*]] = fir.box_addr %[[BOX_F]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:      %[[ATTACH_F:.*]] = acc.attach varPtr(%[[BOX_ADDR_F]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "f"}
! CHECK:      %[[BOX_G:.*]] = fir.load %[[DECLG]]#1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:      %[[BOX_ADDR_G:.*]] = fir.box_addr %[[BOX_G]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:      %[[ATTACH_G:.*]] = acc.attach varPtr(%[[BOX_ADDR_G]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "g"}
! CHECK:      acc.parallel dataOperands(%[[ATTACH_F]], %[[ATTACH_G]] : !fir.ptr<f32>, !fir.ptr<f32>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop private(a) firstprivate(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[ACC_PRIVATE_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.parallel firstprivate(@firstprivatization_section_ext10_ref_10xf32 -> %[[ACC_PRIVATE_B]] : !fir.ref<!fir.array<10xf32>>) {
! CHECK:      %[[ACC_PRIVATE_A:.*]] = acc.private varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:        acc.loop private(@privatization_ref_10xf32 -> %[[ACC_PRIVATE_A]] : !fir.ref<!fir.array<10xf32>>) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop seq
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {seq}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop auto
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop independent
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {independent}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop gang
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop gang {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop gang(num: 8)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        [[GANGNUM1:%.*]] = arith.constant 8 : i32
! CHECK-NEXT:   acc.loop gang(num=[[GANGNUM1]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop gang(num: gangNum)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        [[GANGNUM2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-NEXT:   acc.loop gang(num=[[GANGNUM2]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc parallel loop gang(num: gangNum, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop gang(num=%{{.*}} : i32, static=%{{.*}} : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop vector
  DO i = 1, n
    a(i) = b(i)
  END DO
! CHECK:      acc.parallel {
! CHECK:        acc.loop vector {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop vector(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        [[CONSTANT128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop vector([[CONSTANT128]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop vector(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        [[VECTORLENGTH:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:        acc.loop vector([[VECTORLENGTH]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop worker
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop worker {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop worker(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        [[WORKER128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop worker([[WORKER128]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop collapse(2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:            fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {collapse = 2 : i64}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:            acc.loop {
! CHECK:              fir.do_loop
! CHECK:              acc.yield
! CHECK-NEXT:     }{{$}}
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc parallel loop tile(2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        [[TILESIZE:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop tile([[TILESIZE]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc parallel loop tile(*)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        [[TILESIZEM1:%.*]] = arith.constant -1 : i32
! CHECK:        acc.loop tile([[TILESIZEM1]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop tile(2, 2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.parallel {
! CHECK:        [[TILESIZE1:%.*]] = arith.constant 2 : i32
! CHECK:        [[TILESIZE2:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop tile([[TILESIZE1]], [[TILESIZE2]] : i32, i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop tile(tileSize)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop tile(%{{.*}} : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop tile(tileSize, tileSize)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.parallel {
! CHECK:        acc.loop tile(%{{.*}}, %{{.*}} : i32, i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc parallel loop reduction(+:reduction_r) reduction(*:reduction_i)
  do i = 1, n
    reduction_r = reduction_r + a(i)
    reduction_i = 1
  end do

! CHECK:      %[[COPYINREDR:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) -> !fir.ref<f32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_r"}
! CHECK:      %[[COPYINREDI:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_i"}
! CHECK:      acc.parallel dataOperands(%[[COPYINREDR]], %[[COPYINREDI]] : !fir.ref<f32>, !fir.ref<i32>) {
! CHECK:        acc.loop reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYINREDR]] : !fir.ref<f32>) to varPtr(%{{.*}} : !fir.ref<f32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_r"}
! CHECK:      acc.copyout accPtr(%[[COPYINREDI]] : !fir.ref<i32>) to varPtr(%{{.*}} : !fir.ref<i32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_i"}

  !$acc parallel loop
  do 10 i=0, n
  10 continue
! CHECK: acc.parallel
! CHECK: acc.loop
! CHECK: fir.do_loop

end subroutine acc_parallel_loop

! This test checks lowering of Openacc serial loop combined directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: acc.private.recipe @privatization_ref_10xf32 : !fir.ref<!fir.array<10xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<10xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<10xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK: }

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_10xf32 : !fir.ref<!fir.array<10xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<10xf32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<10xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK: } copy {
! CHECK:  ^bb0(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<!fir.array<10xf32>>):
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: func.func @_QPacc_serial_loop()

subroutine acc_serial_loop
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

  !$acc serial
  !$acc loop
  DO i = 1, n
    a(i) = b(i)
  END DO
  !$acc end serial

! CHECK:      acc.serial {
! CHECK:        acc.loop private{{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial combined(loop) {
! CHECK:        acc.loop combined(serial) private{{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop async
  DO i = 1, n
    a(i) = b(i)
  END DO
  !$acc end serial loop

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {asyncOnly = [#acc.device_type<none>]}

  !$acc serial loop async(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.serial {{.*}} async([[ASYNC1]] : i32) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop async(async)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.serial {{.*}} async([[ASYNC2]] : i32) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop wait
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} wait {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }

  !$acc serial loop wait(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.serial {{.*}} wait({[[WAIT1]] : i32}) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop wait(1, 2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT2:%.*]] = arith.constant 1 : i32
! CHECK:      [[WAIT3:%.*]] = arith.constant 2 : i32
! CHECK:      acc.serial {{.*}} wait({[[WAIT2]] : i32, [[WAIT3]] : i32}) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop wait(wait1, wait2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT4:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      [[WAIT5:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.serial {{.*}} wait({[[WAIT4]] : i32, [[WAIT5]] : i32}) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop if(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[IF1:%.*]] = arith.constant true
! CHECK:      acc.serial {{.*}} if([[IF1]]) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop if(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:      [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK:      acc.serial {{.*}} if([[IF2]]) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop self(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[SELF1:%.*]] = arith.constant true
! CHECK:      acc.serial {{.*}} self([[SELF1]]) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop self
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {selfAttr}

  !$acc serial loop self(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[SELF2:.*]] = fir.convert %[[DECLIFCONDITION]]#1 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:      acc.serial {{.*}} self(%[[SELF2]]) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop copy(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc serial loop copy(a) copy(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc serial loop copyin(a) copyin(readonly: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copyin>, name = "a"}
! CHECK:      acc.delete accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}

  !$acc serial loop copyout(a) copyout(zero: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[CREATE_A]], %[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) {name = "a"}
! CHECK:      acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) {name = "b"}

  !$acc serial loop create(b) create(zero: a)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "a"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[CREATE_B]], %[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_create>, name = "b"}
! CHECK:      acc.delete accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_create_zero>, name = "a"}

  !$acc serial loop no_create(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[NOCREATE_A]], %[[NOCREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop present(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[PRESENT_B:.*]] = acc.present varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[PRESENT_A]], %[[PRESENT_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop deviceptr(a) deviceptr(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop attach(f, g)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[ATTACH_F:.*]] = acc.attach varPtr(%[[DECLF]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>> {name = "f"}
! CHECK:      %[[ATTACH_G:.*]] = acc.attach varPtr(%[[DECLG]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>> {name = "g"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[ATTACH_F]], %[[ATTACH_G]] : !fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop private(a) firstprivate(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[ACC_FPRIVATE_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.serial {{.*}} firstprivate(@firstprivatization_ref_10xf32 -> %[[ACC_FPRIVATE_B]] : !fir.ref<!fir.array<10xf32>>) {
! CHECK:      %[[ACC_PRIVATE_A:.*]] = acc.private varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:        acc.loop {{.*}} private({{.*}}@privatization_ref_10xf32 -> %[[ACC_PRIVATE_A]] : !fir.ref<!fir.array<10xf32>>)
! CHECK-NOT:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop seq
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop auto
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop independent
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop gang
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} gang {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {inclusiveUpperbound = array<i1: true>}{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop gang(num: 8)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        [[GANGNUM1:%.*]] = arith.constant 8 : i32
! CHECK-NEXT:   acc.loop {{.*}} gang({num=[[GANGNUM1]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop gang(num: gangNum)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        [[GANGNUM2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-NEXT:   acc.loop {{.*}} gang({num=[[GANGNUM2]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc serial loop gang(num: gangNum, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} gang({num=%{{.*}} : i32, static=%{{.*}} : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop vector
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} vector {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {inclusiveUpperbound = array<i1: true>}{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop vector(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        [[CONSTANT128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop {{.*}} vector([[CONSTANT128]] : i32) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop vector(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        [[VECTORLENGTH:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:        acc.loop {{.*}} vector([[VECTORLENGTH]] : i32) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop worker
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} worker {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {inclusiveUpperbound = array<i1: true>}{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop worker(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        [[WORKER128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop {{.*}} worker([[WORKER128]] : i32) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop collapse(2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK-NOT:            fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {collapse = [2], collapseDeviceType = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true>}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:            acc.loop {{.*}} {
! CHECK:              acc.yield
! CHECK-NEXT:     }{{$}}
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc serial loop tile(2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        [[TILESIZE:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop {{.*}} tile({[[TILESIZE]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc serial loop tile(*)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        [[TILESIZEM1:%.*]] = arith.constant -1 : i32
! CHECK:        acc.loop {{.*}} tile({[[TILESIZEM1]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop tile(2, 2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        [[TILESIZE1:%.*]] = arith.constant 2 : i32
! CHECK:        [[TILESIZE2:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop {{.*}} tile({[[TILESIZE1]] : i32, [[TILESIZE2]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop tile(tileSize)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} tile({%{{.*}} : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop tile(tileSize, tileSize)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.serial {{.*}} {
! CHECK:        acc.loop {{.*}} tile({%{{.*}} : i32, %{{.*}} : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop reduction(+:reduction_r) reduction(*:reduction_i)
  do i = 1, n
    reduction_r = reduction_r + a(i)
    reduction_i = 1
  end do

! CHECK:      %[[COPYINREDR:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) -> !fir.ref<f32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_r"}
! CHECK:      %[[COPYINREDI:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_i"}
! CHECK:      acc.serial {{.*}} dataOperands(%[[COPYINREDR]], %[[COPYINREDI]] : !fir.ref<f32>, !fir.ref<i32>) {
! CHECK:        acc.loop {{.*}} reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>)
! CHECK-NOT:      fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYINREDR]] : !fir.ref<f32>) to varPtr(%{{.*}} : !fir.ref<f32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_r"}
! CHECK:      acc.copyout accPtr(%[[COPYINREDI]] : !fir.ref<i32>) to varPtr(%{{.*}} : !fir.ref<i32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_i"}

end subroutine acc_serial_loop

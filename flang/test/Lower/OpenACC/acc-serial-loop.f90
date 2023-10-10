! This test checks lowering of Openacc serial loop combined directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_section_ext10_ref_10xf32 : !fir.ref<!fir.array<10xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<10xf32>>):
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<10xf32>
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<10xf32>>
! CHECK: } copy {
! CHECK:  ^bb0(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<!fir.array<10xf32>>):
! CHECK:   acc.terminator
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_10xf32 : !fir.ref<!fir.array<10xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<10xf32>>):
! HLFIR:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! HLFIR:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.private.init"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! HLFIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<10xf32>>
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
! HLFIR: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[B:.*]] = fir.alloca !fir.array<10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
! HLFIR: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca !fir.array<10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
! HLFIR: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]
! CHECK: %[[F:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "f", uniq_name = "{{.*}}Ef"}
! HLFIR: %[[DECLF:.*]]:2 = hlfir.declare %[[F]]
! CHECK: %[[G:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "g", uniq_name = "{{.*}}Eg"}
! HLFIR: %[[DECLG:.*]]:2 = hlfir.declare %[[G]]
! CHECK: %[[IFCONDITION:.*]] = fir.address_of(@{{.*}}ifcondition) : !fir.ref<!fir.logical<4>>
! HLFIR: %[[DECLIFCONDITION:.*]]:2 = hlfir.declare %[[IFCONDITION]]

  !$acc serial loop
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop async
  DO i = 1, n
    a(i) = b(i)
  END DO
  !$acc end serial loop

! CHECK:      acc.serial {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {asyncAttr}

  !$acc serial loop async(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.serial async([[ASYNC1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop async(async)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.serial async([[ASYNC2]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop wait
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {waitAttr}

  !$acc serial loop wait(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.serial wait([[WAIT1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
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
! CHECK:      acc.serial wait([[WAIT2]], [[WAIT3]] : i32, i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
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
! CHECK:      acc.serial wait([[WAIT4]], [[WAIT5]] : i32, i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop if(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[IF1:%.*]] = arith.constant true
! CHECK:      acc.serial if([[IF1]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
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
! CHECK:      acc.serial if([[IF2]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop self(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[SELF1:%.*]] = arith.constant true
! CHECK:      acc.serial self([[SELF1]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop self
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {selfAttr}

  !$acc serial loop self(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[SELF2:.*]] = fir.convert %[[IFCONDITION]] : (!fir.ref<!fir.logical<4>>) -> i1
! HLFIR:      %[[SELF2:.*]] = fir.convert %[[DECLIFCONDITION]]#1 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:      acc.serial self(%[[SELF2]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop copy(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.serial dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! FIR:        acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc serial loop copy(a) copy(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.serial dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! FIR:        acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc serial loop copyin(a) copyin(readonly: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! CHECK:      acc.serial dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop copyout(a) copyout(zero: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! HLFIR:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! FIR:        %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! HLFIR:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! CHECK:      acc.serial dataOperands(%[[CREATE_A]], %[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! FIR:        acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) {name = "a"}
! HLFIR:      acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {name = "a"}
! FIR:        acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) {name = "b"}
! HLFIR:      acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {name = "b"}

  !$acc serial loop create(b) create(zero: a)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! FIR:        %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "a"}
! HLFIR:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "a"}
! CHECK:      acc.serial dataOperands(%[[CREATE_B]], %[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_create>, name = "b"}
! CHECK:      acc.delete accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_create_zero>, name = "a"}

  !$acc serial loop no_create(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.serial dataOperands(%[[NOCREATE_A]], %[[NOCREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop present(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[PRESENT_A:.*]] = acc.present varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[PRESENT_B:.*]] = acc.present varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[PRESENT_B:.*]] = acc.present varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.serial dataOperands(%[[PRESENT_A]], %[[PRESENT_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop deviceptr(a) deviceptr(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.serial dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop attach(f, g)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[BOX_F:.*]] = fir.load %[[F]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! HLFIR:      %[[BOX_F:.*]] = fir.load %[[DECLF]]#1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:      %[[BOX_ADDR_F:.*]] = fir.box_addr %[[BOX_F]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:      %[[ATTACH_F:.*]] = acc.attach varPtr(%[[BOX_ADDR_F]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "f"}
! FIR:        %[[BOX_G:.*]] = fir.load %[[G]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! HLFIR:      %[[BOX_G:.*]] = fir.load %[[DECLG]]#1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:      %[[BOX_ADDR_G:.*]] = fir.box_addr %[[BOX_G]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:      %[[ATTACH_G:.*]] = acc.attach varPtr(%[[BOX_ADDR_G]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "g"}
! CHECK:      acc.serial dataOperands(%[[ATTACH_F]], %[[ATTACH_G]] : !fir.ptr<f32>, !fir.ptr<f32>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop private(a) firstprivate(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[ACC_PRIVATE_A:.*]] = acc.private varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[ACC_PRIVATE_A:.*]] = acc.private varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[ACC_FPRIVATE_B:.*]] = acc.firstprivate varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[ACC_FPRIVATE_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.serial firstprivate(@firstprivatization_section_ext10_ref_10xf32 -> %[[ACC_FPRIVATE_B]] : !fir.ref<!fir.array<10xf32>>) private(@privatization_ref_10xf32 -> %[[ACC_PRIVATE_A]] : !fir.ref<!fir.array<10xf32>>) {
! FIR:        %[[ACC_PRIVATE_A:.*]] = acc.private varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[ACC_PRIVATE_A:.*]] = acc.private varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:        acc.loop private(@privatization_ref_10xf32 -> %[[ACC_PRIVATE_A]] : !fir.ref<!fir.array<10xf32>>) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop seq
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {seq}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop auto
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop independent
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {independent}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop gang
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop gang {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop gang(num: 8)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        [[GANGNUM1:%.*]] = arith.constant 8 : i32
! CHECK-NEXT:   acc.loop gang(num=[[GANGNUM1]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop gang(num: gangNum)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        [[GANGNUM2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-NEXT:   acc.loop gang(num=[[GANGNUM2]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc serial loop gang(num: gangNum, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop gang(num=%{{.*}} : i32, static=%{{.*}} : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop vector
  DO i = 1, n
    a(i) = b(i)
  END DO
! CHECK:      acc.serial {
! CHECK:        acc.loop vector {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop vector(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        [[CONSTANT128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop vector([[CONSTANT128]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop vector(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        [[VECTORLENGTH:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:        acc.loop vector([[VECTORLENGTH]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop worker
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop worker {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop worker(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        [[WORKER128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop worker([[WORKER128]] : i32) {
! CHECK:          fir.do_loop
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

! CHECK:      acc.serial {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:            fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {collapse = 2 : i64}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.serial {
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

 !$acc serial loop tile(2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        [[TILESIZE:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop tile([[TILESIZE]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc serial loop tile(*)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        [[TILESIZEM1:%.*]] = arith.constant -1 : i32
! CHECK:        acc.loop tile([[TILESIZEM1]] : i32) {
! CHECK:          fir.do_loop
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

! CHECK:      acc.serial {
! CHECK:        [[TILESIZE1:%.*]] = arith.constant 2 : i32
! CHECK:        [[TILESIZE2:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop tile([[TILESIZE1]], [[TILESIZE2]] : i32, i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop tile(tileSize)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.serial {
! CHECK:        acc.loop tile(%{{.*}} : i32) {
! CHECK:          fir.do_loop
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

! CHECK:      acc.serial {
! CHECK:        acc.loop tile(%{{.*}}, %{{.*}} : i32, i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc serial loop reduction(+:reduction_r) reduction(*:reduction_i)
  do i = 1, n
    reduction_r = reduction_r + a(i)
    reduction_i = 1
  end do

! CHECK:      acc.serial reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:        acc.loop reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

end subroutine acc_serial_loop

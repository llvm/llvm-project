! This test checks lowering of OpenACC kernels loop combined directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

subroutine acc_kernels_loop
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

  !$acc kernels loop
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop async
  DO i = 1, n
    a(i) = b(i)
  END DO
  !$acc end kernels loop

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: } attributes {asyncAttr}

  !$acc kernels loop async(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels async([[ASYNC1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop async(async)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels async([[ASYNC2]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop wait
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: } attributes {waitAttr}

  !$acc kernels loop wait(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels wait([[WAIT1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop wait(1, 2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT2:%.*]] = arith.constant 1 : i32
! CHECK:      [[WAIT3:%.*]] = arith.constant 2 : i32
! CHECK:      acc.kernels wait([[WAIT2]], [[WAIT3]] : i32, i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop wait(wait1, wait2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT4:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      [[WAIT5:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels wait([[WAIT4]], [[WAIT5]] : i32, i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop num_gangs(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMGANGS1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels num_gangs([[NUMGANGS1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop num_gangs(numGangs)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMGANGS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels num_gangs([[NUMGANGS2]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop num_workers(10)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMWORKERS1:%.*]] = arith.constant 10 : i32
! CHECK:      acc.kernels num_workers([[NUMWORKERS1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop num_workers(numWorkers)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMWORKERS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels num_workers([[NUMWORKERS2]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector_length(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[VECTORLENGTH1:%.*]] = arith.constant 128 : i32
! CHECK:      acc.kernels vector_length([[VECTORLENGTH1]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector_length(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[VECTORLENGTH2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels vector_length([[VECTORLENGTH2]] : i32) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop if(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[IF1:%.*]] = arith.constant true
! CHECK:      acc.kernels if([[IF1]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop if(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:      [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK:      acc.kernels if([[IF2]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop self(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[SELF1:%.*]] = arith.constant true
! CHECK:      acc.kernels self([[SELF1]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop self
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: } attributes {selfAttr}

  !$acc kernels loop self(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[SELF2:.*]] = fir.convert %[[IFCONDITION]] : (!fir.ref<!fir.logical<4>>) -> i1
! HLFIR:      %[[SELF2:.*]] = fir.convert %[[DECLIFCONDITION]]#1 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:      acc.kernels self(%[[SELF2]]) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop copy(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.kernels dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! FIR:        acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc kernels loop copy(a) copy(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.kernels dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! FIR:        acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc kernels loop copyin(a) copyin(readonly: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! CHECK:      acc.kernels dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop copyout(a) copyout(zero: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! HLFIR:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! FIR:        %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! HLFIR:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! CHECK:      acc.kernels dataOperands(%[[CREATE_A]], %[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! FIR:        acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) {name = "a"}
! HLFIR:      acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) {name = "a"}
! FIR:        acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) {name = "b"}
! HLFIR:      acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) {name = "b"}

  !$acc kernels loop create(b) create(zero: a)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! FIR:        %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "a"}
! HLFIR:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "a"}
! CHECK:      acc.kernels dataOperands(%[[CREATE_B]], %[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_create>, name = "b"}
! CHECK:      acc.delete accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_create_zero>, name = "a"}

  !$acc kernels loop no_create(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.kernels dataOperands(%[[NOCREATE_A]], %[[NOCREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop present(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[PRESENT_A:.*]] = acc.present varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[PRESENT_B:.*]] = acc.present varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[PRESENT_B:.*]] = acc.present varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.kernels dataOperands(%[[PRESENT_A]], %[[PRESENT_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop deviceptr(a) deviceptr(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! FIR:        %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! HLFIR:      %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! FIR:        %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[B]] : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! HLFIR:      %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.kernels dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop attach(f, g)
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
! CHECK:      acc.kernels dataOperands(%[[ATTACH_F]], %[[ATTACH_G]] : !fir.ptr<f32>, !fir.ptr<f32>) {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop seq
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {seq}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop auto
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop independent
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {independent}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop gang
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop gang {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop gang(num: 8)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        [[GANGNUM1:%.*]] = arith.constant 8 : i32
! CHECK-NEXT:   acc.loop gang(num=[[GANGNUM1]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop gang(num: gangNum)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        [[GANGNUM2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-NEXT:   acc.loop gang(num=[[GANGNUM2]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

 !$acc kernels loop gang(num: gangNum, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop gang(num=%{{.*}} : i32, static=%{{.*}} : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector
  DO i = 1, n
    a(i) = b(i)
  END DO
! CHECK:      acc.kernels {
! CHECK:        acc.loop vector {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        [[CONSTANT128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop vector([[CONSTANT128]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        [[VECTORLENGTH:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:        acc.loop vector([[VECTORLENGTH]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop worker
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop worker {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop worker(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        [[WORKER128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop worker([[WORKER128]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop collapse(2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:            fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {collapse = 2 : i64}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop {
! CHECK:          fir.do_loop
! CHECK:            acc.loop {
! CHECK:              fir.do_loop
! CHECK:              acc.yield
! CHECK-NEXT:     }{{$}}
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

 !$acc kernels loop tile(2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        [[TILESIZE:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop tile([[TILESIZE]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

 !$acc kernels loop tile(*)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        [[TILESIZEM1:%.*]] = arith.constant -1 : i32
! CHECK:        acc.loop tile([[TILESIZEM1]] : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop tile(2, 2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.kernels {
! CHECK:        [[TILESIZE1:%.*]] = arith.constant 2 : i32
! CHECK:        [[TILESIZE2:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop tile([[TILESIZE1]], [[TILESIZE2]] : i32, i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop tile(tileSize)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop tile(%{{.*}} : i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop tile(tileSize, tileSize)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.kernels {
! CHECK:        acc.loop tile(%{{.*}}, %{{.*}} : i32, i32) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop reduction(+:reduction_r) reduction(*:reduction_i)
  do i = 1, n
    reduction_r = reduction_r + a(i)
    reduction_i = 1
  end do

! CHECK:      acc.kernels {
! CHECK:        acc.loop reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:          fir.do_loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

end subroutine

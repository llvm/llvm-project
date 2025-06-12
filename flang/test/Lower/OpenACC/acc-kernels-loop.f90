! This test checks lowering of OpenACC kernels loop combined directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

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

  !$acc kernels
  !$acc loop
  DO i = 1, n
    a(i) = b(i)
  END DO
  !$acc end kernels

! CHECK:      acc.kernels {
! CHECK:        acc.loop private{{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>]{{.*}}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels combined(loop) {
! CHECK:        acc.loop combined(kernels) private{{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>]{{.*}}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop async
  DO i = 1, n
    a(i) = b(i)
  END DO
  !$acc end kernels loop

! CHECK:      acc.kernels {{.*}} async {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }

  !$acc kernels loop async(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels {{.*}} async([[ASYNC1]] : i32) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop async(async)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels {{.*}} async([[ASYNC2]] : i32) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop async(async) device_type(nvidia) async(1)
  DO i = 1, n
    a(i) = b(i)
  END DO
! CHECK: acc.kernels combined(loop) async(%{{.*}} : i32, %c1{{.*}} : i32 [#acc.device_type<nvidia>])

  !$acc kernels loop wait
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} wait {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }

  !$acc kernels loop wait(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels {{.*}} wait({[[WAIT1]] : i32}) {
! CHECK:        acc.loop
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
! CHECK:      acc.kernels {{.*}} wait({[[WAIT2]] : i32, [[WAIT3]] : i32}) {
! CHECK:        acc.loop
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
! CHECK:      acc.kernels {{.*}} wait({[[WAIT4]] : i32, [[WAIT5]] : i32}) {
! CHECK:        acc.loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop num_gangs(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMGANGS1:%.*]] = arith.constant 1 : i32
! CHECK:      acc.kernels {{.*}} num_gangs({[[NUMGANGS1]] : i32}) {
! CHECK:        acc.loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop num_gangs(numGangs)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMGANGS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels {{.*}} num_gangs({[[NUMGANGS2]] : i32}) {
! CHECK:        acc.loop
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop num_workers(10)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMWORKERS1:%.*]] = arith.constant 10 : i32
! CHECK:      acc.kernels {{.*}} num_workers([[NUMWORKERS1]] : i32) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop num_workers(numWorkers)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[NUMWORKERS2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels {{.*}} num_workers([[NUMWORKERS2]] : i32) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector_length(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[VECTORLENGTH1:%.*]] = arith.constant 128 : i32
! CHECK:      acc.kernels {{.*}} vector_length([[VECTORLENGTH1]] : i32) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector_length(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[VECTORLENGTH2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.kernels {{.*}} vector_length([[VECTORLENGTH2]] : i32) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop if(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[IF1:%.*]] = arith.constant true
! CHECK:      acc.kernels {{.*}} if([[IF1]]) {
! CHECK:        acc.loop {{.*}} {
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
! CHECK:      acc.kernels {{.*}} if([[IF2]]) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop self(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[SELF1:%.*]] = arith.constant true
! CHECK:      acc.kernels {{.*}} self([[SELF1]]) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop self
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}}{
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: } attributes {selfAttr}

  !$acc kernels loop self(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO


! CHECK:      %[[SELF2:.*]] = fir.convert %[[DECLIFCONDITION]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
! CHECK:      acc.kernels {{.*}} self(%[[SELF2]]) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop copy(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO


! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc kernels loop copy(a) copy(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}

  !$acc kernels loop copyin(a) copyin(readonly: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copyin>, name = "a"}
! CHECK:      acc.delete accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}

  !$acc kernels loop copyout(a) copyout(zero: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[CREATE_A]], %[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) {name = "a"}
! CHECK:      acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) {name = "b"}

  !$acc kernels loop create(b) create(zero: a)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "a"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[CREATE_B]], %[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_create>, name = "b"}
! CHECK:      acc.delete accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_create_zero>, name = "a"}

  !$acc kernels loop no_create(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[NOCREATE_A:.*]] = acc.nocreate varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[NOCREATE_B:.*]] = acc.nocreate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[NOCREATE_A]], %[[NOCREATE_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[NOCREATE_A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_no_create>, name = "a"}
! CHECK:      acc.delete accPtr(%[[NOCREATE_B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_no_create>, name = "b"}

  !$acc kernels loop present(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[PRESENT_B:.*]] = acc.present varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[PRESENT_A]], %[[PRESENT_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.delete accPtr(%[[PRESENT_A]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_present>, name = "a"}
! CHECK:      acc.delete accPtr(%[[PRESENT_B]] : !fir.ref<!fir.array<10xf32>>) {dataClause = #acc<data_clause acc_present>, name = "b"}

  !$acc kernels loop deviceptr(a) deviceptr(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[DEVICEPTR_A:.*]] = acc.deviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "a"}
! CHECK:      %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[DEVICEPTR_A]], %[[DEVICEPTR_B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop attach(f, g)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      %[[ATTACH_F:.*]] = acc.attach varPtr(%[[DECLF]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>> {name = "f"}
! CHECK:      %[[ATTACH_G:.*]] = acc.attach varPtr(%[[DECLG]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>> {name = "g"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[ATTACH_F]], %[[ATTACH_G]] : !fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>) {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop seq
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop auto
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop independent
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop gang
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} gang {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop gang(num: 8)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        [[GANGNUM1:%.*]] = arith.constant 8 : i32
! CHECK:        acc.loop {{.*}} gang({num=[[GANGNUM1]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop gang(num: gangNum)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        [[GANGNUM2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:        acc.loop {{.*}} gang({num=[[GANGNUM2]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

 !$acc kernels loop gang(num: gangNum, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} gang({num=%{{.*}} : i32, static=%{{.*}} : i32})
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} vector {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        [[CONSTANT128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop {{.*}} vector([[CONSTANT128]] : i32) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop vector(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        [[VECTORLENGTH:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:        acc.loop {{.*}} vector([[VECTORLENGTH]] : i32) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop worker
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} worker {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop worker(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        [[WORKER128:%.*]] = arith.constant 128 : i32
! CHECK:        acc.loop {{.*}} worker([[WORKER128]] : i32) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop collapse(2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {{{.*}}collapse = [2], collapseDeviceType = [#acc.device_type<none>]{{.*}}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} {
! CHECK:            acc.loop {{.*}} {
! CHECK:              acc.yield
! CHECK-NEXT:     } attributes {auto_ = [#acc.device_type<none>]{{.*}}}
! CHECK:          acc.yield
! CHECK-NEXT:   } attributes {auto_ = [#acc.device_type<none>]{{.*}}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

 !$acc kernels loop tile(2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        [[TILESIZE:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop {{.*}} tile({[[TILESIZE]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

 !$acc kernels loop tile(*)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        [[TILESIZEM1:%.*]] = arith.constant -1 : i32
! CHECK:        acc.loop {{.*}} tile({[[TILESIZEM1]] : i32}) {{.*}} {
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

! CHECK:      acc.kernels {{.*}} {
! CHECK:        [[TILESIZE1:%.*]] = arith.constant 2 : i32
! CHECK:        [[TILESIZE2:%.*]] = arith.constant 2 : i32
! CHECK:        acc.loop {{.*}} tile({[[TILESIZE1]] : i32, [[TILESIZE2]] : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop tile(tileSize)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} tile({%{{.*}} : i32}) {{.*}} {
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

! CHECK:      acc.kernels {{.*}} {
! CHECK:        acc.loop {{.*}} tile({%{{.*}} : i32, %{{.*}} : i32}) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc kernels loop reduction(+:reduction_r) reduction(*:reduction_i)
  do i = 1, n
    reduction_r = reduction_r + a(i)
    reduction_i = 1
  end do

! CHECK:      %[[COPYINREDR:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) -> !fir.ref<f32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_r"}
! CHECK:      %[[COPYINREDI:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_i"}
! CHECK:      acc.kernels {{.*}} dataOperands(%[[COPYINREDR]], %[[COPYINREDI]] : !fir.ref<f32>, !fir.ref<i32>) {
! CHECK:        acc.loop {{.*}} reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>) {{.*}} {
! CHECK:          acc.yield
! CHECK-NEXT:   }{{$}}
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK:      acc.copyout accPtr(%[[COPYINREDR]] : !fir.ref<f32>) to varPtr(%{{.*}} : !fir.ref<f32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_r"}
! CHECK:      acc.copyout accPtr(%[[COPYINREDI]] : !fir.ref<i32>) to varPtr(%{{.*}} : !fir.ref<i32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = "reduction_i"}

end subroutine

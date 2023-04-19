! This test checks lowering of Openacc serial loop combined directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

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

  integer :: gangNum = 8
  integer :: gangStatic = 8
  integer :: vectorNum = 128
  integer, parameter :: tileSize = 2

!CHECK: [[A:%.*]] = fir.alloca !fir.array<10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
!CHECK: [[B:%.*]] = fir.alloca !fir.array<10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
!CHECK: [[C:%.*]] = fir.alloca !fir.array<10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
!CHECK: [[F:%.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "f", uniq_name = "{{.*}}Ef"}
!CHECK: [[G:%.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "g", uniq_name = "{{.*}}Eg"}
!CHECK: [[IFCONDITION:%.*]] = fir.address_of(@{{.*}}ifcondition) : !fir.ref<!fir.logical<4>>

  !$acc serial loop
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop async
  DO i = 1, n
    a(i) = b(i)
  END DO
  !$acc end serial loop

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: } attributes {asyncAttr}

  !$acc serial loop async(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[ASYNC1:%.*]] = arith.constant 1 : i32
!CHECK:      acc.serial async([[ASYNC1]] : i32) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop async(async)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK:      acc.serial async([[ASYNC2]] : i32) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop wait
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: } attributes {waitAttr}

  !$acc serial loop wait(1)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[WAIT1:%.*]] = arith.constant 1 : i32
!CHECK:      acc.serial wait([[WAIT1]] : i32) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop wait(1, 2)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[WAIT2:%.*]] = arith.constant 1 : i32
!CHECK:      [[WAIT3:%.*]] = arith.constant 2 : i32
!CHECK:      acc.serial wait([[WAIT2]], [[WAIT3]] : i32, i32) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop wait(wait1, wait2)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[WAIT4:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK:      [[WAIT5:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK:      acc.serial wait([[WAIT4]], [[WAIT5]] : i32, i32) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop if(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[IF1:%.*]] = arith.constant true
!CHECK:      acc.serial if([[IF1]]) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop if(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK:      [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK:      acc.serial if([[IF2]]) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop self(.TRUE.)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[SELF1:%.*]] = arith.constant true
!CHECK:      acc.serial self([[SELF1]]) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop self
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: } attributes {selfAttr}

  !$acc serial loop self(ifCondition)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[SELF2:%.*]] = fir.convert [[IFCONDITION]] : (!fir.ref<!fir.logical<4>>) -> i1
!CHECK:      acc.serial self([[SELF2]]) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop copy(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial copy([[A]], [[B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop copy(a) copy(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial copy([[A]], [[B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop copyin(a) copyin(readonly: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial copyin([[A]] : !fir.ref<!fir.array<10xf32>>) copyin_readonly([[B]] : !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop copyout(a) copyout(zero: b)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial copyout([[A]] : !fir.ref<!fir.array<10xf32>>) copyout_zero([[B]] : !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop create(b) create(zero: a)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial create([[B]] : !fir.ref<!fir.array<10xf32>>) create_zero([[A]] : !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop no_create(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial no_create([[A]], [[B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop present(a, b)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial present([[A]], [[B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop deviceptr(a) deviceptr(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial deviceptr([[A]], [[B]] : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop attach(f, g)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial attach([[F]], [[G]] : !fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>) {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop private(a) firstprivate(b)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial firstprivate([[B]] : !fir.ref<!fir.array<10xf32>>) private([[A]] : !fir.ref<!fir.array<10xf32>>) {
!CHECK:        acc.loop private([[A]]: !fir.ref<!fir.array<10xf32>>) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop seq
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   } attributes {seq}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop auto
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   } attributes {auto}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop independent
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   } attributes {independent}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop gang
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop gang {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop gang(num: 8)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        [[GANGNUM1:%.*]] = arith.constant 8 : i32
!CHECK-NEXT:   acc.loop gang(num=[[GANGNUM1]]: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop gang(num: gangNum)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        [[GANGNUM2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK-NEXT:   acc.loop gang(num=[[GANGNUM2]]: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

 !$acc serial loop gang(num: gangNum, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop gang(num=%{{.*}}: i32, static=%{{.*}}: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop vector
  DO i = 1, n
    a(i) = b(i)
  END DO
!CHECK:      acc.serial {
!CHECK:        acc.loop vector {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop vector(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        [[CONSTANT128:%.*]] = arith.constant 128 : i32
!CHECK:        acc.loop vector([[CONSTANT128]]: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop vector(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        [[VECTORLENGTH:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK:        acc.loop vector([[VECTORLENGTH]]: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop worker
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop worker {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop worker(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        [[WORKER128:%.*]] = arith.constant 128 : i32
!CHECK:        acc.loop worker([[WORKER128]]: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop collapse(2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:            fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   } attributes {collapse = 2 : i64}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop {
!CHECK:          fir.do_loop
!CHECK:            acc.loop {
!CHECK:              fir.do_loop
!CHECK:              acc.yield
!CHECK-NEXT:     }{{$}}
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

 !$acc serial loop tile(2)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        [[TILESIZE:%.*]] = arith.constant 2 : i32
!CHECK:        acc.loop tile([[TILESIZE]]: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

 !$acc serial loop tile(*)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        [[TILESIZEM1:%.*]] = arith.constant -1 : i32
!CHECK:        acc.loop tile([[TILESIZEM1]]: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop tile(2, 2)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

!CHECK:      acc.serial {
!CHECK:        [[TILESIZE1:%.*]] = arith.constant 2 : i32
!CHECK:        [[TILESIZE2:%.*]] = arith.constant 2 : i32
!CHECK:        acc.loop tile([[TILESIZE1]]: i32, [[TILESIZE2]]: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop tile(tileSize)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop tile(%{{.*}}: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc serial loop tile(tileSize, tileSize)
  DO i = 1, n
    DO j = 1, n
      d(i, j) = e(i, j)
    END DO
  END DO

!CHECK:      acc.serial {
!CHECK:        acc.loop tile(%{{.*}}: i32, %{{.*}}: i32) {
!CHECK:          fir.do_loop
!CHECK:          acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

end subroutine acc_serial_loop

! This test checks lowering of OpenACC loop directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: acc.private.recipe @privatization_ref_10x10xf32 : !fir.ref<!fir.array<10x10xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<10x10xf32>>):
! CHECK: acc.yield %{{.*}} : !fir.ref<!fir.array<10x10xf32>>
! CHECK: }

program acc_loop

  integer :: i, j
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  real, dimension(n, n) :: c, d
  integer :: gangNum = 8
  integer :: gangDim = 1
  integer :: gangStatic = 8
  integer :: vectorLength = 128
  integer, parameter :: tileSize = 2
  integer :: reduction_i
  real :: reduction_r


  !$acc loop
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

 !$acc loop seq
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: } attributes {seq}

  !$acc loop auto
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: } attributes {auto}

  !$acc loop independent
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: } attributes {independent}

  !$acc loop gang
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop gang {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop gang(num: 8)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[GANGNUM1:%.*]] = arith.constant 8 : i32
!CHECK-NEXT: acc.loop gang(num=[[GANGNUM1]] : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop gang(num: gangNum)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[GANGNUM2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK-NEXT: acc.loop gang(num=[[GANGNUM2]] : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

 !$acc loop gang(num: gangNum, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK: acc.loop gang(num=%{{.*}} : i32, static=%{{.*}} : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop vector
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop vector {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop vector(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK: [[CONSTANT128:%.*]] = arith.constant 128 : i32
!CHECK:      acc.loop vector([[CONSTANT128]] : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop vector(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[VECTORLENGTH:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK:      acc.loop vector([[VECTORLENGTH]] : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

!$acc loop worker
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop worker {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop worker(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK: [[WORKER128:%.*]] = arith.constant 128 : i32
!CHECK:      acc.loop worker([[WORKER128]] : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop private(c)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop private(@privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop private(c, d)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop private(@privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>, @privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop private(c) private(d)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop private(@privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>, @privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop tile(2)
  DO i = 1, n
    a(i) = b(i)
  END DO
!CHECK:      [[TILESIZE:%.*]] = arith.constant 2 : i32
!CHECK:      acc.loop tile([[TILESIZE]] : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

 !$acc loop tile(*)
  DO i = 1, n
    a(i) = b(i)
  END DO
!CHECK:      [[TILESIZEM1:%.*]] = arith.constant -1 : i32
!CHECK:      acc.loop tile([[TILESIZEM1]] : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop tile(2, 2)
  DO i = 1, n
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

!CHECK:      [[TILESIZE1:%.*]] = arith.constant 2 : i32
!CHECK:      [[TILESIZE2:%.*]] = arith.constant 2 : i32
!CHECK:      acc.loop tile([[TILESIZE1]], [[TILESIZE2]] : i32, i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop tile(tileSize)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      acc.loop tile(%{{.*}} : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop tile(tileSize, tileSize)
  DO i = 1, n
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

!CHECK:      acc.loop tile(%{{.*}}, %{{.*}} : i32, i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop collapse(2)
  DO i = 1, n
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:          fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: } attributes {collapse = 2 : i64}

  !$acc loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

!CHECK:      acc.loop {
!CHECK:        fir.do_loop
!CHECK:          acc.loop {
!CHECK:            fir.do_loop
!CHECK:            acc.yield
!CHECK-NEXT:   }{{$}}
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop reduction(+:reduction_r) reduction(*:reduction_i)
  do i = 1, n
    reduction_r = reduction_r + a(i)
    reduction_i = 1
  end do

! CHECK:      acc.loop reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:        fir.do_loop
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

 !$acc loop gang(dim: gangDim, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK: acc.loop gang(dim=%{{.*}}, static=%{{.*}} : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop gang(dim: 1)
  DO i = 1, n
    a(i) = b(i)
  END DO

!CHECK:      [[GANGDIM1:%.*]] = arith.constant 1 : i32
!CHECK-NEXT: acc.loop gang(dim=[[GANGDIM1]] : i32) {
!CHECK:        fir.do_loop
!CHECK:        acc.yield
!CHECK-NEXT: }{{$}}

  !$acc loop
  DO i = 1, n
    !$acc cache(b)
    a(i) = b(i)
  END DO

! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK: acc.loop cache(%[[CACHE]] : !fir.ref<!fir.array<10xf32>>)

end program

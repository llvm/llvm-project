! This test checks lowering of OpenACC loop directive.

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

! CHECK: acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}{{$}}

 !$acc loop seq
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK: acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}

  !$acc loop auto
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK: acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

  !$acc loop independent
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

  !$acc loop gang
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop gang() private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop gang(num: 8)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[GANGNUM1:%.*]] = arith.constant 8 : i32
! CHECK-NEXT: acc.loop gang({num=[[GANGNUM1]] : i32}) private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop gang(num: gangNum)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[GANGNUM2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-NEXT: acc.loop gang({num=[[GANGNUM2]] : i32}) private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

 !$acc loop gang(num: gangNum, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK: acc.loop gang({num=%{{.*}} : i32, static=%{{.*}} : i32}) private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop vector
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop vector() private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop vector(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK: [[CONSTANT128:%.*]] = arith.constant 128 : i32
! CHECK:      acc.loop vector([[CONSTANT128]] : i32) private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: }{{$}}

  !$acc loop vector(vectorLength)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[VECTORLENGTH:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:      acc.loop vector([[VECTORLENGTH]] : i32) private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

!$acc loop worker
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop worker() private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop worker(128)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK: [[WORKER128:%.*]] = arith.constant 128 : i32
! CHECK:      acc.loop worker([[WORKER128]] : i32) private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop private(c)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>, @privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop private(c, d)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>, @privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>, @privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop private(c) private(d)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>, @privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>, @privatization_ref_10x10xf32 -> %{{.*}} : !fir.ref<!fir.array<10x10xf32>>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop tile(2)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      [[TILESIZE:%.*]] = arith.constant 2 : i32
! CHECK:      acc.loop {{.*}} tile({[[TILESIZE]] : i32}) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

 !$acc loop tile(*)
  DO i = 1, n
    a(i) = b(i)
  END DO
! CHECK:      [[TILESIZEM1:%.*]] = arith.constant -1 : i32
! CHECK:      acc.loop {{.*}} tile({[[TILESIZEM1]] : i32}) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop tile(2, 2)
  DO i = 1, n
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

! CHECK:      [[TILESIZE1:%.*]] = arith.constant 2 : i32
! CHECK:      [[TILESIZE2:%.*]] = arith.constant 2 : i32
! CHECK:      acc.loop {{.*}} tile({[[TILESIZE1]] : i32, [[TILESIZE2]] : i32}) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop tile(tileSize)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop {{.*}} tile({%{{.*}} : i32}) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop tile(tileSize, tileSize)
  DO i = 1, n
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

! CHECK:      acc.loop {{.*}} tile({%{{.*}} : i32, %{{.*}} : i32}) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop collapse(2)
  DO i = 1, n
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

! CHECK:      acc.loop {{.*}} (%arg0 : i32, %arg1 : i32) = (%{{.*}} : i32, i32) to (%{{.*}} : i32, i32) step (%{{.*}} : i32, i32) {
! CHECK:        fir.store %arg0 to %{{.*}} : !fir.ref<i32>
! CHECK:        fir.store %arg1 to %{{.*}} : !fir.ref<i32>
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {collapse = [2], collapseDeviceType = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true>}

  !$acc loop
  DO i = 1, n
    !$acc loop
    DO j = 1, n
      c(i, j) = d(i, j)
    END DO
  END DO

! CHECK:      acc.loop {{.*}} (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:          acc.loop {{.*}} (%arg1 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:            acc.yield
! CHECK-NEXT:   } attributes {inclusiveUpperbound = array<i1: true>}
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop reduction(+:reduction_r) reduction(*:reduction_i)
  do i = 1, n
    reduction_r = reduction_r + a(i)
    reduction_i = 1
  end do

! CHECK:      acc.loop private(@privatization_ref_i32 -> %{{.*}} : !fir.ref<i32>) reduction(@reduction_add_ref_f32 -> %{{.*}} : !fir.ref<f32>, @reduction_mul_ref_i32 -> %{{.*}} : !fir.ref<i32>) (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

 !$acc loop gang(dim: gangDim, static: gangStatic)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK: acc.loop gang({dim=%{{.*}}, static=%{{.*}} : i32}) {{.*}} (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop gang(dim: 1)
  DO i = 1, n
    a(i) = b(i)
  END DO

! CHECK:      acc.loop gang({dim={{.*}} : i32}) {{.*}} (%arg0 : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:        acc.yield
! CHECK-NEXT: } attributes {inclusiveUpperbound = array<i1: true>}

  !$acc loop
  DO i = 1, n
    !$acc cache(b)
    a(i) = b(i)
  END DO

! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<10xf32>> {name = "b"}
! CHECK: acc.loop {{.*}} cache(%[[CACHE]] : !fir.ref<!fir.array<10xf32>>)

  !$acc loop
  do 100 i=0, n
  100 continue
! CHECK: acc.loop

  !$acc loop gang device_type(nvidia) gang(8)
  DO i = 1, n
  END DO

! CHECK: acc.loop gang([#acc.device_type<none>], {num=%c8{{.*}} : i32} [#acc.device_type<nvidia>])

  !$acc loop device_type(nvidia, default) gang
  DO i = 1, n
  END DO

! CHECK: acc.loop gang([#acc.device_type<nvidia>, #acc.device_type<default>])

end program

subroutine sub1(i, j, k)
  integer :: i,j,k
  integer :: a(i,j,k)
  !$acc parallel loop
  do concurrent (i=1:10,j=1:100,k=1:200)
    a(i,j,k) = a(i,j,k) + 1
  end do
end subroutine

! CHECK: func.func @_QPsub1
! CHECK: %[[DC_K:.*]] = fir.alloca i32 {bindc_name = "k"}
! CHECK: %[[DC_J:.*]] = fir.alloca i32 {bindc_name = "j"}
! CHECK: %[[DC_I:.*]] = fir.alloca i32 {bindc_name = "i"}
! CHECK: acc.parallel
! CHECK: %[[P_I:.*]] = acc.private varPtr(%[[DC_I]] : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = ""}
! CHECK: %[[P_J:.*]] = acc.private varPtr(%[[DC_J]] : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = ""}
! CHECK: %[[P_K:.*]] = acc.private varPtr(%[[DC_K]] : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = ""}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[P_I]] : !fir.ref<i32>, @privatization_ref_i32 -> %[[P_J]] : !fir.ref<i32>, @privatization_ref_i32 -> %[[P_K]] : !fir.ref<i32>) (%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32) = (%c1{{.*}}, %c1{{.*}}, %c1{{.*}} : i32, i32, i32) to (%c10{{.*}}, %c100{{.*}}, %c200{{.*}} : i32, i32, i32)  step (%c1{{.*}}, %c1{{.*}}, %c1{{.*}} : i32, i32, i32)
! CHECK: } attributes {inclusiveUpperbound = array<i1: true, true, true>}

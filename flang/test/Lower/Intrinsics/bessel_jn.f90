! RUN: bbc -emit-fir -hlfir=false %s -o - --math-runtime=fast | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -math-runtime=fast %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir -hlfir=false %s -o - --math-runtime=relaxed | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -math-runtime=relaxed %s -o - | FileCheck --check-prefixes=ALL %s
! RUN: bbc -emit-fir -hlfir=false %s -o - --math-runtime=precise | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -math-runtime=precise %s -o - | FileCheck --check-prefixes=ALL %s

! ALL-LABEL: @_QPtest_real4
! ALL-SAME: (%[[argx:.*]]: !fir.ref<f32>{{.*}}, %[[argn:.*]]: !fir.ref<i32>{{.*}}) -> f32
function test_real4(x, n)
  real :: x, test_real4
  integer :: n

  ! ALL-DAG: %[[x:.*]] = fir.load %[[argx]] : !fir.ref<f32>
  ! ALL-DAG: %[[n:.*]] = fir.load %[[argn]] : !fir.ref<i32>
  ! ALL: fir.call @jnf(%[[n]], %[[x]]) {{.*}} : (i32, f32) -> f32
  test_real4 = bessel_jn(n, x)
end function

! ALL-LABEL: @_QPtest_real8
! ALL-SAME: (%[[argx:.*]]: !fir.ref<f64>{{.*}}, %[[argn:.*]]: !fir.ref<i32>{{.*}}) -> f64
function test_real8(x, n)
  real(8) :: x, test_real8
  integer :: n

  ! ALL-DAG: %[[x:.*]] = fir.load %[[argx]] : !fir.ref<f64>
  ! ALL-DAG: %[[n:.*]] = fir.load %[[argn]] : !fir.ref<i32>
  ! ALL: fir.call @jn(%[[n]], %[[x]]) {{.*}} : (i32, f64) -> f64
  test_real8 = bessel_jn(n, x)
end function

! ALL-LABEL: @_QPtest_transformational_real4
! ALL-SAME: %[[argx:.*]]: !fir.ref<f32>{{.*}}, %[[argn1:.*]]: !fir.ref<i32>{{.*}}, %[[argn2:.*]]: !fir.ref<i32>{{.*}}
subroutine test_transformational_real4(x, n1, n2, r)
  real(4) :: x
  integer :: n1, n2
  real(4) :: r(:)

  ! ALL-DAG: %[[zero:.*]] = arith.constant 0{{.*}} : f32
  ! ALL-DAG: %[[one:.*]] = arith.constant 1 : i32
  ! ALL-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! ALL-DAG: %[[x:.*]] = fir.load %[[argx]] : !fir.ref<f32>
  ! ALL-DAG: %[[n1:.*]] = fir.load %[[argn1]] : !fir.ref<i32>
  ! ALL-DAG: %[[n2:.*]] = fir.load %[[argn2]] : !fir.ref<i32>
  ! ALL-DAG: %[[xeq0:.*]] = arith.cmpf ueq, %[[x]], %[[zero]] {{.*}} : f32
  ! ALL-DAG: %[[n1ltn2:.*]] = arith.cmpi slt, %[[n1]], %[[n2]] : i32
  ! ALL-DAG: %[[n1eqn2:.*]] = arith.cmpi eq, %[[n1]], %[[n2]] : i32
  ! ALL: fir.if %[[xeq0]] {
  ! ALL: %[[resxeq0:.*]] = fir.convert %[[r]] {{.*}}
  ! ALL: fir.call @_FortranABesselJnX0_4(%[[resxeq0]], %[[n1]], %[[n2]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, !fir.ref<i8>, i32) -> ()
  ! ALL-NEXT: } else {
  ! ALL-NEXT: fir.if %[[n1ltn2]] {
  ! ALL-DAG: %[[n2_1:.*]] = arith.subi %[[n2]], %[[one]] : i32
  ! ALL-DAG: %[[bn2:.*]] = fir.call @jnf(%[[n2]], %[[x]]) {{.*}} : (i32, f32) -> f32
  ! ALL-DAG: %[[bn2_1:.*]] = fir.call @jnf(%[[n2_1]], %[[x]]) {{.*}} : (i32, f32) -> f32
  ! ALL-DAG: %[[resn1ltn2:.*]] = fir.convert %[[r]] {{.*}}
  ! ALL: fir.call @_FortranABesselJn_4(%[[resn1ltn2]], %[[n1]], %[[n2]], %[[x]], %[[bn2]], %[[bn2_1]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! ALL-NEXT: } else {
  ! ALL-NEXT: fir.if %[[n1eqn2]] {
  ! ALL-DAG: %[[bn2:.*]] = fir.call @jnf(%[[n2]], %[[x]]) {{.*}} : (i32, f32) -> f32
  ! ALL-DAG: %[[resn1eqn2:.*]] = fir.convert %[[r]] {{.*}}
  ! ALL: fir.call @_FortranABesselJn_4(%[[resn1eqn2]], %[[n1]], %[[n2]], %[[x]], %[[bn2]], %[[zero]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! ALL-NEXT: } else {
  ! ALL-DAG: %[[resn1gtn2:.*]] = fir.convert %[[r]] {{.*}}
  ! ALL: fir.call @_FortranABesselJn_4(%[[resn1gtn2]], %[[n1]], %[[n2]], %[[x]], %[[zero]], %[[zero]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! ALL-NEXT: }
  ! ALL-NEXT: }
  ! ALL-NEXT: }
  r = bessel_jn(n1, n2, x)
  ! ALL: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! ALL: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! ALL: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xf32>>
end subroutine test_transformational_real4

! ALL-LABEL: @_QPtest_transformational_real8
! ALL-SAME: %[[argx:.*]]: !fir.ref<f64>{{.*}}, %[[argn1:.*]]: !fir.ref<i32>{{.*}}, %[[argn2:.*]]: !fir.ref<i32>{{.*}}
subroutine test_transformational_real8(x, n1, n2, r)
  real(8) :: x
  integer :: n1, n2
  real(8) :: r(:)

  ! ALL-DAG: %[[zero:.*]] = arith.constant 0{{.*}} : f64
  ! ALL-DAG: %[[one:.*]] = arith.constant 1 : i32
  ! ALL-DAG: %[[r:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
  ! ALL-DAG: %[[x:.*]] = fir.load %[[argx]] : !fir.ref<f64>
  ! ALL-DAG: %[[n1:.*]] = fir.load %[[argn1]] : !fir.ref<i32>
  ! ALL-DAG: %[[n2:.*]] = fir.load %[[argn2]] : !fir.ref<i32>
  ! ALL-DAG: %[[xeq0:.*]] = arith.cmpf ueq, %[[x]], %[[zero]] {{.*}} : f64
  ! ALL-DAG: %[[n1ltn2:.*]] = arith.cmpi slt, %[[n1]], %[[n2]] : i32
  ! ALL-DAG: %[[n1eqn2:.*]] = arith.cmpi eq, %[[n1]], %[[n2]] : i32
  ! ALL: fir.if %[[xeq0]] {
  ! ALL: %[[resxeq0:.*]] = fir.convert %[[r]] {{.*}}
  ! ALL: fir.call @_FortranABesselJnX0_8(%[[resxeq0]], %[[n1]], %[[n2]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, !fir.ref<i8>, i32) -> ()
  ! ALL-NEXT: } else {
  ! ALL-NEXT: fir.if %[[n1ltn2]] {
  ! ALL-DAG: %[[n2_1:.*]] = arith.subi %[[n2]], %[[one]] : i32
  ! ALL-DAG: %[[bn2:.*]] = fir.call @jn(%[[n2]], %[[x]]) {{.*}} : (i32, f64) -> f64
  ! ALL-DAG: %[[bn2_1:.*]] = fir.call @jn(%[[n2_1]], %[[x]]) {{.*}} : (i32, f64) -> f64
  ! ALL-DAG: %[[resn1ltn2:.*]] = fir.convert %[[r]] {{.*}}
  ! ALL: fir.call @_FortranABesselJn_8(%[[resn1ltn2]], %[[n1]], %[[n2]], %[[x]], %[[bn2]], %[[bn2_1]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! ALL-NEXT: } else {
  ! ALL-NEXT: fir.if %[[n1eqn2]] {
  ! ALL-DAG: %[[bn2:.*]] = fir.call @jn(%[[n2]], %[[x]]) {{.*}} : (i32, f64) -> f64
  ! ALL-DAG: %[[resn1eqn2:.*]] = fir.convert %[[r]] {{.*}}
  ! ALL: fir.call @_FortranABesselJn_8(%[[resn1eqn2]], %[[n1]], %[[n2]], %[[x]], %[[bn2]], %[[zero]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! ALL-NEXT: } else {
  ! ALL-DAG: %[[resn1gtn2:.*]] = fir.convert %[[r]] {{.*}}
  ! ALL: fir.call @_FortranABesselJn_8(%[[resn1gtn2]], %[[n1]], %[[n2]], %[[x]], %[[zero]], %[[zero]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! ALL-NEXT: }
  ! ALL-NEXT: }
  ! ALL-NEXT: }
  r = bessel_jn(n1, n2, x)
  ! ALL: %[[box:.*]] = fir.load %[[r]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  ! ALL: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
  ! ALL: fir.freemem %[[addr]] : !fir.heap<!fir.array<?xf64>>
end subroutine test_transformational_real8

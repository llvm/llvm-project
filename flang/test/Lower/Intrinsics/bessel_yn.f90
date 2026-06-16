! RUN: %flang_fc1 -emit-hlfir %s -o - -mllvm -math-runtime=fast | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-hlfir %s -o - -mllvm -math-runtime=relaxed | FileCheck --check-prefixes=ALL %s
! RUN: %flang_fc1 -emit-hlfir %s -o - -mllvm -math-runtime=precise | FileCheck --check-prefixes=ALL %s

! ALL-LABEL: func.func @_QPtest_real4
! ALL-SAME: (%[[argx:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}, %[[argn:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) -> f32 {
function test_real4(x, n)
  real :: x, test_real4
  integer :: n

  ! ALL: %[[scope:.*]] = fir.dummy_scope : !fir.dscope
  ! ALL: %[[n_decl:.*]]:2 = hlfir.declare %[[argn]] dummy_scope %[[scope]] arg 2 {uniq_name = "_QFtest_real4En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! ALL: %[[res_alloc:.*]] = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  ! ALL: %[[res_decl:.*]]:2 = hlfir.declare %[[res_alloc]] {uniq_name = "_QFtest_real4Etest_real4"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! ALL: %[[x_decl:.*]]:2 = hlfir.declare %[[argx]] dummy_scope %[[scope]] arg 1 {uniq_name = "_QFtest_real4Ex"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! ALL-DAG: %[[n:.*]] = fir.load %[[n_decl]]#0 : !fir.ref<i32>
  ! ALL-DAG: %[[x:.*]] = fir.load %[[x_decl]]#0 : !fir.ref<f32>
  ! ALL: fir.call @ynf(%[[n]], %[[x]]) {{.*}} : (i32, f32) -> f32
  test_real4 = bessel_yn(n, x)
end function

! ALL-LABEL: func.func @_QPtest_real8
! ALL-SAME: (%[[argx:.*]]: !fir.ref<f64> {fir.bindc_name = "x"}, %[[argn:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) -> f64 {
function test_real8(x, n)
  real(8) :: x, test_real8
  integer :: n

  ! ALL: %[[scope:.*]] = fir.dummy_scope : !fir.dscope
  ! ALL: %[[n_decl:.*]]:2 = hlfir.declare %[[argn]] dummy_scope %[[scope]] arg 2 {uniq_name = "_QFtest_real8En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! ALL: %[[res_alloc:.*]] = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  ! ALL: %[[res_decl:.*]]:2 = hlfir.declare %[[res_alloc]] {uniq_name = "_QFtest_real8Etest_real8"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  ! ALL: %[[x_decl:.*]]:2 = hlfir.declare %[[argx]] dummy_scope %[[scope]] arg 1 {uniq_name = "_QFtest_real8Ex"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  ! ALL-DAG: %[[n:.*]] = fir.load %[[n_decl]]#0 : !fir.ref<i32>
  ! ALL-DAG: %[[x:.*]] = fir.load %[[x_decl]]#0 : !fir.ref<f64>
  ! ALL: fir.call @yn(%[[n]], %[[x]]) {{.*}} : (i32, f64) -> f64
  test_real8 = bessel_yn(n, x)
end function

! ALL-LABEL: func.func @_QPtest_transformational_real4
! ALL-SAME: (%[[argx:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}, %[[argn1:.*]]: !fir.ref<i32> {fir.bindc_name = "n1"}, %[[argn2:.*]]: !fir.ref<i32> {fir.bindc_name = "n2"}, %[[argr:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "r"}) {
subroutine test_transformational_real4(x, n1, n2, r)
  real(4) :: x
  integer :: n1, n2
  real(4) :: r(:)

  ! ALL: %[[temp:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! ALL: %[[scope:.*]] = fir.dummy_scope : !fir.dscope
  ! ALL: %[[n1_decl:.*]]:2 = hlfir.declare %[[argn1]] dummy_scope %[[scope]] arg 2 {uniq_name = "_QFtest_transformational_real4En1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! ALL: %[[n2_decl:.*]]:2 = hlfir.declare %[[argn2]] dummy_scope %[[scope]] arg 3 {uniq_name = "_QFtest_transformational_real4En2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! ALL: %[[r_decl:.*]]:2 = hlfir.declare %[[argr]] dummy_scope %[[scope]] arg 4 {uniq_name = "_QFtest_transformational_real4Er"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  ! ALL: %[[x_decl:.*]]:2 = hlfir.declare %[[argx]] dummy_scope %[[scope]] arg 1 {uniq_name = "_QFtest_transformational_real4Ex"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! ALL-DAG: %[[n1:.*]] = fir.load %[[n1_decl]]#0 : !fir.ref<i32>
  ! ALL-DAG: %[[n2:.*]] = fir.load %[[n2_decl]]#0 : !fir.ref<i32>
  ! ALL-DAG: %[[x:.*]] = fir.load %[[x_decl]]#0 : !fir.ref<f32>
  ! ALL: %[[zero:.*]] = arith.constant 0{{.*}} : f32
  ! ALL: %[[one:.*]] = arith.constant 1 : i32
  ! ALL: %[[xeq0:.*]] = arith.cmpf ueq, %[[x]], %[[zero]] {{.*}} : f32
  ! ALL: %[[n1ltn2:.*]] = arith.cmpi slt, %[[n1]], %[[n2]] : i32
  ! ALL: %[[n1eqn2:.*]] = arith.cmpi eq, %[[n1]], %[[n2]] : i32
  ! ALL: fir.if %[[xeq0]] {
  ! ALL: %[[resxeq0:.*]] = fir.convert %[[temp]] {{.*}}
  ! ALL: fir.call @_FortranABesselYnX0_4(%[[resxeq0]], %[[n1]], %[[n2]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, !fir.ref<i8>, i32) -> ()
  ! ALL: } else {
  ! ALL: fir.if %[[n1ltn2]] {
  ! ALL-DAG: %[[n1_1:.*]] = arith.addi %[[n1]], %[[one]] : i32
  ! ALL-DAG: %[[bn1:.*]] = fir.call @ynf(%[[n1]], %[[x]]) {{.*}} : (i32, f32) -> f32
  ! ALL-DAG: %[[bn1_1:.*]] = fir.call @ynf(%[[n1_1]], %[[x]]) {{.*}} : (i32, f32) -> f32
  ! ALL: %[[resn1ltn2:.*]] = fir.convert %[[temp]] {{.*}}
  ! ALL: fir.call @_FortranABesselYn_4(%[[resn1ltn2]], %[[n1]], %[[n2]], %[[x]], %[[bn1]], %[[bn1_1]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! ALL: } else {
  ! ALL: fir.if %[[n1eqn2]] {
  ! ALL-DAG: %[[bn1:.*]] = fir.call @ynf(%[[n1]], %[[x]]) {{.*}} : (i32, f32) -> f32
  ! ALL: %[[resn1eqn2:.*]] = fir.convert %[[temp]] {{.*}}
  ! ALL: fir.call @_FortranABesselYn_4(%[[resn1eqn2]], %[[n1]], %[[n2]], %[[x]], %[[bn1]], %[[zero]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! ALL: } else {
  ! ALL: %[[resn1gtn2:.*]] = fir.convert %[[temp]] {{.*}}
  ! ALL: fir.call @_FortranABesselYn_4(%[[resn1gtn2]], %[[n1]], %[[n2]], %[[x]], %[[zero]], %[[zero]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f32, f32, f32, !fir.ref<i8>, i32) -> ()
  ! ALL: }
  ! ALL: }
  ! ALL: }
  r = bessel_yn(n1, n2, x)
end subroutine test_transformational_real4

! ALL-LABEL: func.func @_QPtest_transformational_real8
! ALL-SAME: (%[[argx:.*]]: !fir.ref<f64> {fir.bindc_name = "x"}, %[[argn1:.*]]: !fir.ref<i32> {fir.bindc_name = "n1"}, %[[argn2:.*]]: !fir.ref<i32> {fir.bindc_name = "n2"}, %[[argr:.*]]: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "r"}) {
subroutine test_transformational_real8(x, n1, n2, r)
  real(8) :: x
  integer :: n1, n2
  real(8) :: r(:)

  ! ALL: %[[temp:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
  ! ALL: %[[scope:.*]] = fir.dummy_scope : !fir.dscope
  ! ALL: %[[n1_decl:.*]]:2 = hlfir.declare %[[argn1]] dummy_scope %[[scope]] arg 2 {uniq_name = "_QFtest_transformational_real8En1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! ALL: %[[n2_decl:.*]]:2 = hlfir.declare %[[argn2]] dummy_scope %[[scope]] arg 3 {uniq_name = "_QFtest_transformational_real8En2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! ALL: %[[r_decl:.*]]:2 = hlfir.declare %[[argr]] dummy_scope %[[scope]] arg 4 {uniq_name = "_QFtest_transformational_real8Er"} : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>)
  ! ALL: %[[x_decl:.*]]:2 = hlfir.declare %[[argx]] dummy_scope %[[scope]] arg 1 {uniq_name = "_QFtest_transformational_real8Ex"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
  ! ALL-DAG: %[[n1:.*]] = fir.load %[[n1_decl]]#0 : !fir.ref<i32>
  ! ALL-DAG: %[[n2:.*]] = fir.load %[[n2_decl]]#0 : !fir.ref<i32>
  ! ALL-DAG: %[[x:.*]] = fir.load %[[x_decl]]#0 : !fir.ref<f64>
  ! ALL: %[[zero:.*]] = arith.constant 0{{.*}} : f64
  ! ALL: %[[one:.*]] = arith.constant 1 : i32
  ! ALL: %[[xeq0:.*]] = arith.cmpf ueq, %[[x]], %[[zero]] {{.*}} : f64
  ! ALL: %[[n1ltn2:.*]] = arith.cmpi slt, %[[n1]], %[[n2]] : i32
  ! ALL: %[[n1eqn2:.*]] = arith.cmpi eq, %[[n1]], %[[n2]] : i32
  ! ALL: fir.if %[[xeq0]] {
  ! ALL: %[[resxeq0:.*]] = fir.convert %[[temp]] {{.*}}
  ! ALL: fir.call @_FortranABesselYnX0_8(%[[resxeq0]], %[[n1]], %[[n2]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, !fir.ref<i8>, i32) -> ()
  ! ALL: } else {
  ! ALL: fir.if %[[n1ltn2]] {
  ! ALL-DAG: %[[n1_1:.*]] = arith.addi %[[n1]], %[[one]] : i32
  ! ALL-DAG: %[[bn1:.*]] = fir.call @yn(%[[n1]], %[[x]]) {{.*}} : (i32, f64) -> f64
  ! ALL-DAG: %[[bn1_1:.*]] = fir.call @yn(%[[n1_1]], %[[x]]) {{.*}} : (i32, f64) -> f64
  ! ALL: %[[resn1ltn2:.*]] = fir.convert %[[temp]] {{.*}}
  ! ALL: fir.call @_FortranABesselYn_8(%[[resn1ltn2]], %[[n1]], %[[n2]], %[[x]], %[[bn1]], %[[bn1_1]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! ALL: } else {
  ! ALL: fir.if %[[n1eqn2]] {
  ! ALL-DAG: %[[bn1:.*]] = fir.call @yn(%[[n1]], %[[x]]) {{.*}} : (i32, f64) -> f64
  ! ALL: %[[resn1eqn2:.*]] = fir.convert %[[temp]] {{.*}}
  ! ALL: fir.call @_FortranABesselYn_8(%[[resn1eqn2]], %[[n1]], %[[n2]], %[[x]], %[[bn1]], %[[zero]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! ALL: } else {
  ! ALL: %[[resn1gtn2:.*]] = fir.convert %[[temp]] {{.*}}
  ! ALL: fir.call @_FortranABesselYn_8(%[[resn1gtn2]], %[[n1]], %[[n2]], %[[x]], %[[zero]], %[[zero]], {{.*}}, {{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i32, i32, f64, f64, f64, !fir.ref<i8>, i32) -> ()
  ! ALL: }
  ! ALL: }
  ! ALL: }
  r = bessel_yn(n1, n2, x)
end subroutine test_transformational_real8

! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: @_QPis_finite_test
subroutine is_finite_test(x, y)
  use ieee_arithmetic, only: ieee_is_finite
  real(4) x
  real(8) y
  ! CHECK:   %[[V_3:[0-9]+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK:   %[[V_4:[0-9]+]] = arith.bitcast %[[V_3]] : f32 to i32
  ! CHECK:   %[[V_5:[0-9]+]] = arith.subi %c32{{.*}}, %c8{{.*}} : i32
  ! CHECK:   %[[V_6:[0-9]+]] = arith.shrui %c-1{{.*}}, %[[V_5]] : i32
  ! CHECK:   %[[V_7:[0-9]+]] = arith.shrsi %[[V_4]], %c23{{.*}} : i32
  ! CHECK:   %[[V_8:[0-9]+]] = arith.andi %[[V_7]], %[[V_6]] : i32
  ! CHECK:   %[[V_9:[0-9]+]] = arith.cmpi eq, %c8{{.*}}, %c0{{.*}} : i32
  ! CHECK:   %[[V_10:[0-9]+]] = arith.select %[[V_9]], %c0{{.*}}, %[[V_8]] : i32
  ! CHECK:   %[[V_11:[0-9]+]] = arith.cmpi ne, %[[V_10]], %c255{{.*}} : i32
  ! CHECK:   %[[V_12:[0-9]+]] = fir.convert %[[V_11]] : (i1) -> !fir.logical<4>
  ! CHECK:   %[[V_13:[0-9]+]] = fir.convert %[[V_12]] : (!fir.logical<4>) -> i1
  print*, ieee_is_finite(x)

  ! CHECK:   %[[V_19:[0-9]+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK:   %[[V_20:[0-9]+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK:   %[[V_21:[0-9]+]] = arith.addf %[[V_19]], %[[V_20]] : f32
  ! CHECK:   %[[V_22:[0-9]+]] = arith.bitcast %[[V_21]] : f32 to i32
  ! CHECK:   %[[V_23:[0-9]+]] = arith.subi %c32{{.*}}, %c8{{.*}} : i32
  ! CHECK:   %[[V_24:[0-9]+]] = arith.shrui %c-1{{.*}}, %[[V_23]] : i32
  ! CHECK:   %[[V_25:[0-9]+]] = arith.shrsi %[[V_22]], %c23{{.*}} : i32
  ! CHECK:   %[[V_26:[0-9]+]] = arith.andi %[[V_25]], %[[V_24]] : i32
  ! CHECK:   %[[V_27:[0-9]+]] = arith.cmpi eq, %c8{{.*}}, %c0{{.*}} : i32
  ! CHECK:   %[[V_28:[0-9]+]] = arith.select %[[V_27]], %c0{{.*}}, %[[V_26]] : i32
  ! CHECK:   %[[V_29:[0-9]+]] = arith.cmpi ne, %[[V_28]], %c255{{.*}} : i32
  ! CHECK:   %[[V_30:[0-9]+]] = fir.convert %[[V_29]] : (i1) -> !fir.logical<4>
  ! CHECK:   %[[V_31:[0-9]+]] = fir.convert %[[V_30]] : (!fir.logical<4>) -> i1
  print*, ieee_is_finite(x+x)

  ! CHECK:   %[[V_37:[0-9]+]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK:   %[[V_38:[0-9]+]] = arith.bitcast %[[V_37]] : f64 to i64
  ! CHECK:   %[[V_39:[0-9]+]] = arith.subi %c64{{.*}}, %c11{{.*}} : i64
  ! CHECK:   %[[V_40:[0-9]+]] = arith.shrui %c-1{{.*}}, %[[V_39]] : i64
  ! CHECK:   %[[V_41:[0-9]+]] = arith.shrsi %[[V_38]], %c52{{.*}} : i64
  ! CHECK:   %[[V_42:[0-9]+]] = arith.andi %[[V_41]], %[[V_40]] : i64
  ! CHECK:   %[[V_43:[0-9]+]] = arith.cmpi eq, %c11{{.*}}, %c0{{.*}} : i64
  ! CHECK:   %[[V_44:[0-9]+]] = arith.select %[[V_43]], %c0{{.*}}, %[[V_42]] : i64
  ! CHECK:   %[[V_45:[0-9]+]] = arith.cmpi ne, %[[V_44]], %c2047{{.*}} : i64
  ! CHECK:   %[[V_46:[0-9]+]] = fir.convert %[[V_45]] : (i1) -> !fir.logical<4>
  ! CHECK:   %[[V_47:[0-9]+]] = fir.convert %[[V_46]] : (!fir.logical<4>) -> i1
  print*, ieee_is_finite(y)

  ! CHECK:   %[[V_53:[0-9]+]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK:   %[[V_54:[0-9]+]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK:   %[[V_55:[0-9]+]] = arith.addf %[[V_53]], %[[V_54]] : f64
  ! CHECK:   %[[V_56:[0-9]+]] = arith.bitcast %[[V_55]] : f64 to i64
  ! CHECK:   %[[V_57:[0-9]+]] = arith.subi %c64{{.*}}, %c11{{.*}} : i64
  ! CHECK:   %[[V_58:[0-9]+]] = arith.shrui %c-1{{.*}}, %[[V_57]] : i64
  ! CHECK:   %[[V_59:[0-9]+]] = arith.shrsi %[[V_56]], %c52{{.*}} : i64
  ! CHECK:   %[[V_60:[0-9]+]] = arith.andi %[[V_59]], %[[V_58]] : i64
  ! CHECK:   %[[V_61:[0-9]+]] = arith.cmpi eq, %c11{{.*}}, %c0{{.*}} : i64
  ! CHECK:   %[[V_62:[0-9]+]] = arith.select %[[V_61]], %c0{{.*}}, %[[V_60]] : i64
  ! CHECK:   %[[V_63:[0-9]+]] = arith.cmpi ne, %[[V_62]], %c2047{{.*}} : i64
  ! CHECK:   %[[V_64:[0-9]+]] = fir.convert %[[V_63]] : (i1) -> !fir.logical<4>
  ! CHECK:   %[[V_65:[0-9]+]] = fir.convert %[[V_64]] : (!fir.logical<4>) -> i1
  print*, ieee_is_finite(y+y)
end subroutine is_finite_test

  real(4) x
  real(8) y
  call is_finite_test(huge(x), huge(y))
end

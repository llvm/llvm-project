! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPkagi
function kagi(index)
  ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca i32 {bindc_name = "kagi"
  ! CHECK:   %[[V_1:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK:   %[[V_2:[0-9]+]] = arith.cmpi slt, %[[V_1]], %c0{{.*}} : i32
  ! CHECK:   cf.cond_br %[[V_2]], ^bb2, ^bb1
  ! CHECK: ^bb1:  // pred: ^bb0
  ! CHECK:   %[[V_3:[0-9]+]] = arith.cmpi sgt, %[[V_1]], %c0{{.*}} : i32
  ! CHECK:   cf.cond_br %[[V_3]], ^bb4, ^bb3
  ! CHECK: ^bb2:  // pred: ^bb0
  ! CHECK:   fir.store %c1{{.*}} to %[[V_0]] : !fir.ref<i32>
  ! CHECK:   cf.br ^bb5
  ! CHECK: ^bb3:  // pred: ^bb1
  ! CHECK:   fir.store %c2{{.*}} to %[[V_0]] : !fir.ref<i32>
  ! CHECK:   cf.br ^bb5
  ! CHECK: ^bb4:  // pred: ^bb1
  ! CHECK:   fir.store %c3{{.*}} to %[[V_0]] : !fir.ref<i32>
  ! CHECK:   cf.br ^bb5
  ! CHECK: ^bb5:  // 3 preds: ^bb2, ^bb3, ^bb4
  ! CHECK:   %[[V_4:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
  ! CHECK:   return %[[V_4]] : i32
  if (index) 7, 8, 9
  kagi = 0; return
7 kagi = 1; return
8 kagi = 2; return
9 kagi = 3; return
end

! CHECK-LABEL: func @_QPkagf
function kagf(findex)
  ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca i32 {bindc_name = "kagf"
  ! CHECK:   %[[V_1:[0-9]+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK:   %[[V_2:[0-9]+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK:   %[[V_3:[0-9]+]] = arith.addf %[[V_1]], %[[V_2]] {{.*}} : f32
  ! CHECK:   %[[V_4:[0-9]+]] = arith.addf %[[V_3]], %[[V_3]] {{.*}} : f32
  ! CHECK:   %cst = arith.constant 0.000000e+00 : f32
  ! CHECK:   %[[V_5:[0-9]+]] = arith.cmpf olt, %[[V_4]], %cst {{.*}} : f32
  ! CHECK:   cf.cond_br %[[V_5]], ^bb2, ^bb1
  ! CHECK: ^bb1:  // pred: ^bb0
  ! CHECK:   %[[V_6:[0-9]+]] = arith.cmpf ogt, %[[V_4]], %cst {{.*}} : f32
  ! CHECK:   cf.cond_br %[[V_6]], ^bb4, ^bb3
  ! CHECK: ^bb2:  // pred: ^bb0
  ! CHECK:   fir.store %c1{{.*}} to %[[V_0]] : !fir.ref<i32>
  ! CHECK:   cf.br ^bb5
  ! CHECK: ^bb3:  // pred: ^bb1
  ! CHECK:   fir.store %c2{{.*}} to %[[V_0]] : !fir.ref<i32>
  ! CHECK:   cf.br ^bb5
  ! CHECK: ^bb4:  // pred: ^bb1
  ! CHECK:   fir.store %c3{{.*}} to %[[V_0]] : !fir.ref<i32>
  ! CHECK:   cf.br ^bb5
  ! CHECK: ^bb5:  // 3 preds: ^bb2, ^bb3, ^bb4
  ! CHECK:   %[[V_7:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
  ! CHECK:   return %[[V_7]] : i32
  if (findex+findex) 7, 8, 9
  kagf = 0; return
7 kagf = 1; return
8 kagf = 2; return
9 kagf = 3; return
end

! CHECK-LABEL: func @_QQmain
  do i = -2, 2
    print*, kagi(i)
  enddo

  print*, kagf(-2.0)
  print*, kagf(-1.0)
  print*, kagf(-0.0)
  print*, kagf( 0.0)
  print*, kagf(+0.0)
  print*, kagf(+1.0)
  print*, kagf(+2.0)
end

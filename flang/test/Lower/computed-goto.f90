! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPm
function m(index)
    ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca i32 {bindc_name = "m"
    ! CHECK:   %[[V_1:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
    ! CHECK:   fir.select %[[V_1]] : i32 [1, ^bb6, 2, ^bb5, 3, ^bb4, 4, ^bb3, 5, ^bb2, unit, ^bb1]
    ! CHECK: ^bb1:  // pred: ^bb0
    ! CHECK:   fir.store %c0{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb2:  // pred: ^bb0
    ! CHECK:   fir.store %c1{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb3:  // pred: ^bb0
    ! CHECK:   fir.store %c3{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb4:  // pred: ^bb0
    ! CHECK:   fir.store %c5{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb5:  // pred: ^bb0
    ! CHECK:   fir.store %c7{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb6:  // pred: ^bb0
    ! CHECK:   fir.store %c9{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb7:  // 6 preds: ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6
    ! CHECK:   %[[V_2:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
    ! CHECK:   return %[[V_2]] : i32
    goto (9,7,5,3,1) index ! + 1
    m = 0; return
1   m = 1; return
3   m = 3; return
5   m = 5; return
7   m = 7; return
9   m = 9; return
end

! CHECK-LABEL: func @_QPm1
function m1(index)
    ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca i32 {bindc_name = "m1"
    ! CHECK:   %[[V_1:[0-9]+]] = llvm.intr.stacksave : !llvm.ptr
    ! CHECK:   %[[V_2:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
    ! CHECK:   %[[V_3:[0-9]+]] = arith.cmpi eq, %[[V_2]], %c1{{.*}} : i32
    ! CHECK:   cf.cond_br %[[V_3]], ^bb1, ^bb2
    ! CHECK: ^bb1:  // pred: ^bb0
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   cf.br ^bb3
    ! CHECK: ^bb2:  // pred: ^bb0
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   fir.store %c0{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb4
    ! CHECK: ^bb3:  // pred: ^bb1
    ! CHECK:   fir.store %c10{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb4
    ! CHECK: ^bb4:  // 2 preds: ^bb2, ^bb3
    ! CHECK:   %[[V_4:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
    ! CHECK:   return %[[V_4]] : i32
    block
      goto (10) index
    end block
    m1 =  0; return
10  m1 = 10; return
end

! CHECK-LABEL: func @_QPm2
function m2(index)
    ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca i32 {bindc_name = "m2"
    ! CHECK:   %[[V_1:[0-9]+]] = llvm.intr.stacksave : !llvm.ptr
    ! CHECK:   %[[V_2:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
    ! CHECK:   %[[V_3:[0-9]+]] = arith.cmpi eq, %[[V_2]], %c1{{.*}} : i32
    ! CHECK:   cf.cond_br %[[V_3]], ^bb1, ^bb2
    ! CHECK: ^bb1:  // pred: ^bb0
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   cf.br ^bb5
    ! CHECK: ^bb2:  // pred: ^bb0
    ! CHECK:   %[[V_4:[0-9]+]] = arith.cmpi eq, %[[V_2]], %c2{{.*}} : i32
    ! CHECK:   cf.cond_br %[[V_4]], ^bb3, ^bb4
    ! CHECK: ^bb3:  // pred: ^bb2
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   cf.br ^bb6
    ! CHECK: ^bb4:  // pred: ^bb2
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   fir.store %c0{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb5:  // pred: ^bb1
    ! CHECK:   fir.store %c10{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb6:  // pred: ^bb3
    ! CHECK:   fir.store %c20{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb7:  // 3 preds: ^bb4, ^bb5, ^bb6
    ! CHECK:   %[[V_5:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
    ! CHECK:   return %[[V_5]] : i32
    block
      goto (10,20) index
    end block
    m2 =  0; return
10  m2 = 10; return
20  m2 = 20; return
end

! CHECK-LABEL: func @_QPm3
function m3(index)
    ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca i32 {bindc_name = "m3"
    ! CHECK:   %[[V_1:[0-9]+]] = llvm.intr.stacksave : !llvm.ptr
    ! CHECK:   %[[V_2:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
    ! CHECK:   %[[V_3:[0-9]+]] = arith.cmpi eq, %[[V_2]], %c1{{.*}} : i32
    ! CHECK:   cf.cond_br %[[V_3]], ^bb1, ^bb2
    ! CHECK: ^bb1:  // pred: ^bb0
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   cf.br ^bb7
    ! CHECK: ^bb2:  // pred: ^bb0
    ! CHECK:   %[[V_4:[0-9]+]] = arith.cmpi eq, %[[V_2]], %c2{{.*}} : i32
    ! CHECK:   cf.cond_br %[[V_4]], ^bb3, ^bb4
    ! CHECK: ^bb3:  // pred: ^bb2
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   cf.br ^bb8
    ! CHECK: ^bb4:  // pred: ^bb2
    ! CHECK:   %[[V_5:[0-9]+]] = arith.cmpi eq, %[[V_2]], %c3{{.*}} : i32
    ! CHECK:   cf.cond_br %[[V_5]], ^bb5, ^bb6
    ! CHECK: ^bb5:  // pred: ^bb4
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   cf.br ^bb9
    ! CHECK: ^bb6:  // pred: ^bb4
    ! CHECK:   llvm.intr.stackrestore %[[V_1]] : !llvm.ptr
    ! CHECK:   fir.store %c0{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb10
    ! CHECK: ^bb7:  // pred: ^bb1
    ! CHECK:   fir.store %c10{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb10
    ! CHECK: ^bb8:  // pred: ^bb3
    ! CHECK:   fir.store %c20{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb10
    ! CHECK: ^bb9:  // pred: ^bb5
    ! CHECK:   fir.store %c30{{.*}} to %[[V_0]] : !fir.ref<i32>
    ! CHECK:   cf.br ^bb10
    ! CHECK: ^bb10:  // 4 preds: ^bb6, ^bb7, ^bb8, ^bb9
    ! CHECK:   %[[V_6:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<i32>
    ! CHECK:   return %[[V_6]] : i32
    block
      goto (10,20,30) index
    end block
    m3 =  0; return
10  m3 = 10; return
20  m3 = 20; return
30  m3 = 30; return
end

program cg
  print*, m(-3), m(1), m(2), m(3), m(4), m(5), m(9) ! 0 9 7 5 3 1 0
  print*, m1(0), m1(1), m1(2) ! 0 10 0
  print*, m2(0), m2(1), m2(2), m2(3) ! 0 10 20 0
  print*, m3(0), m3(1), m3(2), m3(3), m3(4) ! 0 10 20 30 0
end

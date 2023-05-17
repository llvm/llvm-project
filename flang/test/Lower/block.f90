! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
program bb ! block stack management and exits
    ! CHECK:   %[[V_1:[0-9]+]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    integer :: i, j
    ! CHECK:   fir.store %c0{{.*}} to %[[V_1]] : !fir.ref<i32>
    i = 0
    ! CHECK:   %[[V_3:[0-9]+]] = fir.call @llvm.stacksave()
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   br ^bb1
    ! CHECK: ^bb1:  // 2 preds: ^bb0, ^bb15
    ! CHECK:   cond_br %{{.*}}, ^bb2, ^bb16
    ! CHECK: ^bb2:  // pred: ^bb1
    ! CHECK:   %[[V_11:[0-9]+]] = fir.call @llvm.stacksave()
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   cond_br %{{.*}}, ^bb3, ^bb4
    ! CHECK: ^bb3:  // pred: ^bb2
    ! CHECK:   br ^bb10
    ! CHECK: ^bb4:  // pred: ^bb2
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   cond_br %{{.*}}, ^bb5, ^bb6
    ! CHECK: ^bb5:  // pred: ^bb4
    ! CHECK:   br ^bb7
    ! CHECK: ^bb6:  // pred: ^bb4
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   cond_br %{{.*}}, ^bb7, ^bb8
    ! CHECK: ^bb7:  // 3 preds: ^bb5, ^bb6, ^bb12
    ! CHECK:   fir.call @llvm.stackrestore(%[[V_11]])
    ! CHECK:   br ^bb14
    ! CHECK: ^bb8:  // pred: ^bb6
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   cond_br %{{.*}}, ^bb9, ^bb10
    ! CHECK: ^bb9:  // pred: ^bb8
    ! CHECK:   fir.call @llvm.stackrestore(%[[V_11]])
    ! CHECK:   br ^bb15
    ! CHECK: ^bb10:  // 2 preds: ^bb3, ^bb8
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   cond_br %{{.*}}, ^bb11, ^bb12
    ! CHECK: ^bb11:  // pred: ^bb10
    ! CHECK:   fir.call @llvm.stackrestore(%[[V_11]])
    ! CHECK:   br ^bb17
    ! CHECK: ^bb12:  // pred: ^bb10
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   cond_br %{{.*}}, ^bb13, ^bb7
    ! CHECK: ^bb13:  // pred: ^bb12
    ! CHECK:   fir.call @llvm.stackrestore(%[[V_11]])
    ! CHECK:   fir.call @llvm.stackrestore(%[[V_3]])
    ! CHECK:   br ^bb18
    ! CHECK: ^bb14:  // pred: ^bb7
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   br ^bb15
    ! CHECK: ^bb15:  // 2 preds: ^bb9, ^bb14
    ! CHECK:   br ^bb1
    ! CHECK: ^bb16:  // pred: ^bb1
    ! CHECK:   fir.store %{{.*}} to %[[V_1]] : !fir.ref<i32>
    ! CHECK:   br ^bb17
    ! CHECK: ^bb17:  // 2 preds: ^bb11, ^bb16
    ! CHECK:   fir.call @llvm.stackrestore(%[[V_3]])
    ! CHECK:   br ^bb18
    ! CHECK: ^bb18:  // 2 preds: ^bb13, ^bb17
    ! CHECK:   return
    block
      i = i + 1 ! 1 increment
      do j = 1, 5
        block
          i = i + 1; if (j == 1) goto 1   ! inner block - 5 increments, 1 goto
          i = i + 1; if (j == 2) goto 2   ! inner block - 4 increments, 1 goto
          i = i + 1; if (j == 3) goto 10  ! outer block - 3 increments, 1 goto
          i = i + 1; if (j == 4) goto 11  ! outer block - 2 increments, 1 goto
1         i = i + 1; if (j == 5) goto 12  ! outer block - 2 increments, 1 goto
          i = i + 1; if (j == 6) goto 100 ! program     - 1 increment
2       end block
10      i = i + 1 ! 3 increments
11    end do
      i = i + 1 ! 0 increments
12  end block
100 print*, i ! expect 21
end

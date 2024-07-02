!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Checks lowering of OpenMP variables with implicitly determined DSAs.

! Basic cases.
!CHECK-LABEL: func @_QPimplicit_dsa_test1
!CHECK:       %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test1Ex"}
!CHECK:       %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_test1Ey"}
!CHECK:       %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFimplicit_dsa_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFimplicit_dsa_test1Ez"}
!CHECK:       %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFimplicit_dsa_test1Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       omp.task {
!CHECK-NEXT:    %[[PRIV_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFimplicit_dsa_test1Ey"}
!CHECK-NEXT:    %[[PRIV_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV_Y]] {uniq_name = "_QFimplicit_dsa_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test1Ex"}
!CHECK-NEXT:    %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:    hlfir.assign %[[TEMP]] to %[[PRIV_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NOT:     fir.alloca
!CHECK:       }
!CHECK:       omp.task {
!CHECK-NOT:     fir.alloca
!CHECK:       }
subroutine implicit_dsa_test1
  integer :: x, y, z

  !$omp task private(y) shared(z)
    x = y + z
  !$omp end task

  !$omp task default(shared)
    x = y + z
  !$omp end task
end subroutine

! Nested task with implicit firstprivate DSA variable.
!CHECK-LABEL: func @_QPimplicit_dsa_test2
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test2Ex"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.task {
!CHECK:   omp.task {
!CHECK:     %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test2Ex"}
!CHECK:     %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:     %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK:     hlfir.assign %[[TEMP]] to %[[PRIV_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:   }
!CHECK: }
subroutine implicit_dsa_test2
  integer :: x

  !$omp task
    !$omp task
      x = 1
    !$omp end task
  !$omp end task
end subroutine

! Nested tasks with implicit shared DSA variables.
!CHECK-LABEL: func @_QPimplicit_dsa_test3
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test3Ex"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_test3Ey"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFimplicit_dsa_test3Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFimplicit_dsa_test3Ez"}
!CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFimplicit_dsa_test3Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel {
!CHECK:   omp.task {
!CHECK:     %[[ONE:.*]] = arith.constant 1 : i32
!CHECK:     hlfir.assign %[[ONE]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:     %[[ONE:.*]] = arith.constant 1 : i32
!CHECK:     hlfir.assign %[[ONE]] to %[[Y_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:   }
!CHECK:   omp.task {
!CHECK:     %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test3Ex"}
!CHECK:     %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:     %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK:     hlfir.assign %[[TEMP]] to %[[PRIV_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:     %[[ONE:.*]] = arith.constant 1 : i32
!CHECK:     hlfir.assign %[[ONE]] to %[[PRIV_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:     %[[ONE:.*]] = arith.constant 1 : i32
!CHECK:     hlfir.assign %[[ONE]] to %[[Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:   }
!CHECK: }
subroutine implicit_dsa_test3
  integer :: x, y, z

  !$omp parallel
    !$omp task
      x = 1
      y = 1
    !$omp end task

    !$omp task firstprivate(x)
      x = 1
      z = 1
    !$omp end task
  !$omp end parallel
end subroutine

! Task with implicit firstprivate DSA variables, enclosed in private context.
!CHECK-LABEL: func @_QPimplicit_dsa_test4
!CHECK:       %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test4Ex"}
!CHECK:       %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_test4Ey"}
!CHECK:       %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFimplicit_dsa_test4Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFimplicit_dsa_test4Ez"}
!CHECK:       %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFimplicit_dsa_test4Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       omp.parallel {
!CHECK:         %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test4Ex"}
!CHECK:         %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         %[[PRIV_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFimplicit_dsa_test4Ez"}
!CHECK:         %[[PRIV_Z_DECL:.*]]:2 = hlfir.declare %[[PRIV_Z]] {uniq_name = "_QFimplicit_dsa_test4Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         %[[PRIV_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFimplicit_dsa_test4Ey"}
!CHECK:         %[[PRIV_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV_Y]] {uniq_name = "_QFimplicit_dsa_test4Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         omp.task {
!CHECK-NEXT:      %[[PRIV2_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test4Ex"}
!CHECK-NEXT:      %[[PRIV2_X_DECL:.*]]:2 = hlfir.declare %[[PRIV2_X]] {uniq_name = "_QFimplicit_dsa_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[TEMP:.*]] = fir.load %[[PRIV_X_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP]] to %[[PRIV2_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:      %[[PRIV2_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFimplicit_dsa_test4Ez"}
!CHECK-NEXT:      %[[PRIV2_Z_DECL:.*]]:2 = hlfir.declare %[[PRIV2_Z]] {uniq_name = "_QFimplicit_dsa_test4Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[TEMP2:.*]] = fir.load %[[PRIV_Z_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP2]] to %[[PRIV2_Z_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
!CHECK-NEXT:      hlfir.assign %[[ZERO]] to %[[PRIV2_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:           %[[ONE:.*]] = arith.constant 1 : i32
!CHECK-NEXT:      hlfir.assign %[[ONE]] to %[[PRIV2_Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:         }
!CHECK:         omp.task {
!CHECK-NEXT:      %[[PRIV2_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test4Ex"}
!CHECK-NEXT:      %[[PRIV2_X_DECL:.*]]:2 = hlfir.declare %[[PRIV2_X]] {uniq_name = "_QFimplicit_dsa_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[TEMP:.*]] = fir.load %[[PRIV_X_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP]] to %[[PRIV2_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:      %[[PRIV2_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFimplicit_dsa_test4Ey"}
!CHECK-NEXT:      %[[PRIV2_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV2_Y]] {uniq_name = "_QFimplicit_dsa_test4Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[TEMP2:.*]] = fir.load %[[PRIV_Y_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP2]] to %[[PRIV2_Y_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:           %[[ONE:.*]] = arith.constant 1 : i32
!CHECK-NEXT:      hlfir.assign %[[ONE]] to %[[PRIV2_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
!CHECK-NEXT:      hlfir.assign %[[ZERO]] to %[[PRIV2_Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:         }
!CHECK:       }
subroutine implicit_dsa_test4
  integer :: x, y, z

  !$omp parallel default(private)
    !$omp task
      x = 0
      z = 1
    !$omp end task

    !$omp task
      x = 1
      y = 0
    !$omp end task
  !$omp end parallel
end subroutine

! Inner parallel using implicit firstprivate symbol.
!CHECK-LABEL: func @_QPimplicit_dsa_test5
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test5Ex"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test5Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel {
!CHECK:     %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test5Ex"}
!CHECK:     %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test5Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   omp.task {
!CHECK:     %[[PRIV2_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test5Ex"}
!CHECK:     %[[PRIV2_X_DECL:.*]]:2 = hlfir.declare %[[PRIV2_X]] {uniq_name = "_QFimplicit_dsa_test5Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[TEMP:.*]] = fir.load %[[PRIV_X_DECL]]#0 : !fir.ref<i32>
!CHECK:      hlfir.assign %[[TEMP]] to %[[PRIV2_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:     omp.parallel {
!CHECK:       %[[ONE:.*]] = arith.constant 1 : i32
!CHECK:       hlfir.assign %[[ONE]] to %[[PRIV2_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:     }
!CHECK:   }
!CHECK: }
subroutine implicit_dsa_test5
  integer :: x

  !$omp parallel default(private)
    !$omp task
      !$omp parallel
        x = 1
      !$omp end parallel
    !$omp end task
  !$omp end parallel
end subroutine

! Constructs nested inside a task with implicit DSA variables.
!CHECK-LABEL: func @_QPimplicit_dsa_test6
!CHECK:       %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test6Ex"}
!CHECK:       %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test6Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_test6Ey"}
!CHECK:       %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFimplicit_dsa_test6Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFimplicit_dsa_test6Ez"}
!CHECK:       %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFimplicit_dsa_test6Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       omp.task {
!CHECK-NEXT:    %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test6Ex"}
!CHECK-NEXT:    %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test6Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:    hlfir.assign %[[TEMP]] to %[[PRIV_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:    %[[PRIV_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFimplicit_dsa_test6Ey"}
!CHECK-NEXT:    %[[PRIV_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV_Y]] {uniq_name = "_QFimplicit_dsa_test6Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[TEMP2:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:    hlfir.assign %[[TEMP2]] to %[[PRIV_Y_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:    %[[PRIV_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFimplicit_dsa_test6Ez"}
!CHECK-NEXT:    %[[PRIV_Z_DECL:.*]]:2 = hlfir.declare %[[PRIV_Z]] {uniq_name = "_QFimplicit_dsa_test6Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[TEMP3:.*]] = fir.load %[[Z_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:    hlfir.assign %[[TEMP3]] to %[[PRIV_Z_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:         omp.parallel {
!CHECK:           %[[PRIV2_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test6Ex"}
!CHECK:           %[[PRIV2_X_DECL:.*]]:2 = hlfir.declare %[[PRIV2_X]] {uniq_name = "_QFimplicit_dsa_test6Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NOT:       hlfir.assign
!CHECK:           %[[PRIV2_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFimplicit_dsa_test6Ey"}
!CHECK:           %[[PRIV2_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV2_Y]] {uniq_name = "_QFimplicit_dsa_test6Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NOT:       hlfir.assign
!CHECK:           hlfir.assign %{{.*}} to %[[PRIV2_X_DECL]]
!CHECK:         }
!CHECK:         omp.parallel {
!CHECK-NEXT:      %[[PRIV3_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test6Ex"}
!CHECK-NEXT:      %[[PRIV3_X_DECL:.*]]:2 = hlfir.declare %[[PRIV3_X]] {uniq_name = "_QFimplicit_dsa_test6Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[TEMP4:.*]] = fir.load %[[PRIV_X_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP4]] to %[[PRIV3_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:      %[[PRIV3_Z:.*]] = fir.alloca i32 {bindc_name = "z", pinned, uniq_name = "_QFimplicit_dsa_test6Ez"}
!CHECK-NEXT:      %[[PRIV3_Z_DECL:.*]]:2 = hlfir.declare %[[PRIV3_Z]] {uniq_name = "_QFimplicit_dsa_test6Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[TEMP5:.*]] = fir.load %[[PRIV_Z_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP5]] to %[[PRIV3_Z_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:           hlfir.assign %{{.*}} to %[[PRIV_Y_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:         }
!CHECK:       }
subroutine implicit_dsa_test6
  integer :: x, y, z

  !$omp task
    !$omp parallel default(private)
      x = y
    !$omp end parallel

    !$omp parallel default(firstprivate) shared(y)
      y = x + z
    !$omp end parallel
  !$omp end task
end subroutine

! Test taskgroup - it uses the same scope as task.
!CHECK-LABEL: func @_QPimplicit_dsa_test7
!CHECK:       %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test7Ex"}
!CHECK:       %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test7Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_test7Ey"}
!CHECK:       %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFimplicit_dsa_test7Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       omp.task {
!CHECK:         omp.taskgroup {
!CHECK-NEXT:      %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFimplicit_dsa_test7Ex"}
!CHECK-NEXT:      %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test7Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[TEMP:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP]] to %[[PRIV_X_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK-NEXT:      %[[PRIV_Y:.*]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFimplicit_dsa_test7Ey"}
!CHECK-NEXT:      %[[PRIV_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV_Y]] {uniq_name = "_QFimplicit_dsa_test7Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[TEMP2:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP2]] to %[[PRIV_Y_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:         }
!CHECK:       }
subroutine implicit_dsa_test7
  integer :: x, y

  !$omp task
    !$omp taskgroup
      x = y
    !$omp end taskgroup
  !$omp end task
end subroutine

! TODO Test taskloop

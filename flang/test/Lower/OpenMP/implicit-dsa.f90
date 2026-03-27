! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 \
! RUN: | FileCheck %s

! Checks lowering of OpenMP variables with implicitly determined DSAs.

! Privatizers

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TASKLOOP_TEST3_I_PRIVATE:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TASKLOOP_TEST3_X_FIRSTPRIVATE:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TASKLOOP_TEST2_X_PRIVATE:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TASKLOOP_TEST2_I_PRIVATE:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TASKLOOP_TEST1_I_PRIVATE:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TASKLOOP_TEST1_X_FIRSTPRIVATE:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TASKLOOP_TEST1_Y_PRIVATE:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST7_Y_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST7_X_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TEST6_Y_PRIV:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TEST6_X_PRIV:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST6_Z_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST6_Y_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST6_X_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST5_X_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TEST5_X_PRIV:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST4_Y_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST4_Z_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST4_X_FIRSTPRIV:.*]] : i32
! CHECK-SAME:  copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TEST4_Y_PRIV:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TEST4_Z_PRIV:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TEST4_X_PRIV:.*]] : i32
! CHECK-NOT:   copy {

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST3_X_FIRSTPRIV:.*]] : i32
! CHECK:       copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST2_X_FIRSTPRIV:.*]] : i32
! CHECK:       copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = firstprivate} @[[TEST1_X_FIRSTPRIV:.*]] : i32
! CHECK:       copy {
! CHECK:         hlfir.assign

! CHECK-LABEL: omp.private
! CHECK-SAME:      {type = private} @[[TEST1_Y_PRIV:.*]] : i32
! CHECK-NOT:   copy {

! Basic cases.
!CHECK-LABEL: func @_QPimplicit_dsa_test1
!CHECK:       %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test1Ex"}
!CHECK:       %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_test1Ey"}
!CHECK:       %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFimplicit_dsa_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFimplicit_dsa_test1Ez"}
!CHECK:       %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFimplicit_dsa_test1Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       omp.task private(@[[TEST1_Y_PRIV]] %[[Y_DECL]]#0 -> %[[PRIV_Y:.*]], @[[TEST1_X_FIRSTPRIV]] %[[X_DECL]]#0 -> %[[PRIV_X:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT:    %[[PRIV_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV_Y]] {uniq_name = "_QFimplicit_dsa_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       }
!CHECK:       omp.task {
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
!CHECK:   omp.task private(@[[TEST2_X_FIRSTPRIV]] %[[X_DECL]]#0 -> %[[PRIV_X:.*]] : !fir.ref<i32>) {
!CHECK:     %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
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
!CHECK:   omp.task private(@[[TEST3_X_FIRSTPRIV]] %[[X_DECL]]#0 -> %[[PRIV_X]] : !fir.ref<i32>) {
!CHECK:     %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
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
!CHECK:       omp.parallel private({{.*}} %{{.*}}#0 -> %[[PRIV_X:.*]], {{.*}} %{{.*}}#0 -> %[[PRIV_Z:.*]], {{.*}} %{{.*}}#0 -> %[[PRIV_Y:.*]] : {{.*}}) {
!CHECK:         %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         %[[PRIV_Z_DECL:.*]]:2 = hlfir.declare %[[PRIV_Z]] {uniq_name = "_QFimplicit_dsa_test4Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         %[[PRIV_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV_Y]] {uniq_name = "_QFimplicit_dsa_test4Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         omp.task private(@[[TEST4_X_FIRSTPRIV]] %[[PRIV_X_DECL]]#0 -> %[[PRIV2_X:.*]], @[[TEST4_Z_FIRSTPRIV]] %[[PRIV_Z_DECL]]#0 -> %[[PRIV2_Z:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT:      %[[PRIV2_X_DECL:.*]]:2 = hlfir.declare %[[PRIV2_X]] {uniq_name = "_QFimplicit_dsa_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[PRIV2_Z_DECL:.*]]:2 = hlfir.declare %[[PRIV2_Z]] {uniq_name = "_QFimplicit_dsa_test4Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[ZERO:.*]] = arith.constant 0 : i32
!CHECK-NEXT:      hlfir.assign %[[ZERO]] to %[[PRIV2_X_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:           %[[ONE:.*]] = arith.constant 1 : i32
!CHECK-NEXT:      hlfir.assign %[[ONE]] to %[[PRIV2_Z_DECL]]#0 : i32, !fir.ref<i32>
!CHECK:         }
!CHECK:         omp.task private(@[[TEST4_X_FIRSTPRIV]] %[[PRIV_X_DECL]]#0 -> %[[PRIV2_X:.*]], @[[TEST4_Y_FIRSTPRIV]] %[[PRIV_Y_DECL]]#0 -> %[[PRIV2_Y:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT:      %[[PRIV2_X_DECL:.*]]:2 = hlfir.declare %[[PRIV2_X]] {uniq_name = "_QFimplicit_dsa_test4Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[PRIV2_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV2_Y]] {uniq_name = "_QFimplicit_dsa_test4Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
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
!CHECK: omp.parallel private({{.*}} %{{.*}}#0 -> %[[PRIV_X:.*]] : {{.*}}) {
!CHECK:     %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test5Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:   omp.task private(@[[TEST5_X_FIRSTPRIV]] %[[PRIV_X_DECL]]#0 -> %[[PRIV2_X:.*]] : !fir.ref<i32>) {
!CHECK:     %[[PRIV2_X_DECL:.*]]:2 = hlfir.declare %[[PRIV2_X]] {uniq_name = "_QFimplicit_dsa_test5Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
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
!CHECK:       omp.task private(@[[TEST6_X_FIRSTPRIV]] %[[X_DECL]]#0 -> %[[PRIV_X:.*]], @[[TEST6_Y_FIRSTPRIV]] %[[Y_DECL]]#0 -> %[[PRIV_Y:.*]], @[[TEST6_Z_FIRSTPRIV]] %[[Z_DECL]]#0 -> %[[PRIV_Z:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT:    %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test6Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[PRIV_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV_Y]] {uniq_name = "_QFimplicit_dsa_test6Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:    %[[PRIV_Z_DECL:.*]]:2 = hlfir.declare %[[PRIV_Z]] {uniq_name = "_QFimplicit_dsa_test6Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         omp.parallel private({{.*}} %{{.*}}#0 -> %[[PRIV2_X:.*]], {{.*}} %{{.*}}#0 -> %[[PRIV2_Y:.*]] : {{.*}}) {
!CHECK:           %[[PRIV2_X_DECL:.*]]:2 = hlfir.declare %[[PRIV2_X]] {uniq_name = "_QFimplicit_dsa_test6Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NOT:       hlfir.assign
!CHECK:           %[[PRIV2_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV2_Y]] {uniq_name = "_QFimplicit_dsa_test6Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NOT:       hlfir.assign
!CHECK:           hlfir.assign %{{.*}} to %[[PRIV2_X_DECL]]
!CHECK:         }
!CHECK:         omp.parallel private({{.*firstprivate.*}} %{{.*}}#0 -> %[[PRIV3_X:.*]], {{.*firstprivate.*}} %{{.*}}#0 -> %[[PRIV3_Z:.*]] : {{.*}}) {
!CHECK-NEXT:      %[[PRIV3_X_DECL:.*]]:2 = hlfir.declare %[[PRIV3_X]] {uniq_name = "_QFimplicit_dsa_test6Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-NEXT:      %[[PRIV3_Z_DECL:.*]]:2 = hlfir.declare %[[PRIV3_Z]] {uniq_name = "_QFimplicit_dsa_test6Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
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

! Test taskgroup.
!CHECK-LABEL: func @_QPimplicit_dsa_test7
!CHECK:       %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_test7Ex"}
!CHECK:       %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_test7Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_test7Ey"}
!CHECK:       %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFimplicit_dsa_test7Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:       omp.task private(@[[TEST7_X_FIRSTPRIV]] %[[X_DECL]]#0 -> %[[PRIV_X:[^,]*]],
!CHECK-SAME:      @[[TEST7_Y_FIRSTPRIV]] %[[Y_DECL]]#0 -> %[[PRIV_Y:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK:         %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFimplicit_dsa_test7Ex"}
!CHECK:         %[[PRIV_Y_DECL:.*]]:2 = hlfir.declare %[[PRIV_Y]] {uniq_name = "_QFimplicit_dsa_test7Ey"}
!CHECK:         omp.taskgroup {
!CHECK-NEXT:      %[[TEMP:.*]] = fir.load %[[PRIV_Y_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT:      hlfir.assign %[[TEMP]] to %[[PRIV_X_DECL]]#0 : i32, !fir.ref<i32>
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

! Test taskloop
! CHECK-LABEL:   func.func @_QPimplicit_dsa_taskloop_test1
! CHECK:           %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFimplicit_dsa_taskloop_test1Ei"}
! CHECK:           %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFimplicit_dsa_taskloop_test1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_taskloop_test1Ex"}
! CHECK:           %[[DECL_X:.*]]:2 = hlfir.declare %[[ALLOCA_X]] {uniq_name = "_QFimplicit_dsa_taskloop_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_taskloop_test1Ey"}
! CHECK:           %[[DECL_Y:.*]]:2 = hlfir.declare %[[ALLOCA_Y]] {uniq_name = "_QFimplicit_dsa_taskloop_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFimplicit_dsa_taskloop_test1Ez"}
! CHECK:           %[[DECL_Z:.*]]:2 = hlfir.declare %[[ALLOCA_Z]] {uniq_name = "_QFimplicit_dsa_taskloop_test1Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine implicit_dsa_taskloop_test1
   integer :: x, y, z
   ! CHECK: omp.taskloop private(
   ! CHECK-SAME: @[[TASKLOOP_TEST1_Y_PRIVATE]] %[[DECL_Y]]#0 -> %[[ARG0:.*]], @[[TASKLOOP_TEST1_X_FIRSTPRIVATE]] %[[DECL_X]]#0 -> %[[ARG1:.*]], @[[TASKLOOP_TEST1_I_PRIVATE]] %[[DECL_I]]#0 -> %[[ARG2:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
   ! CHECK: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
   !$omp taskloop private(y) shared(z)
   do i = 1, 100
      ! CHECK: %[[Y_VAL:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFimplicit_dsa_taskloop_test1Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      ! CHECK: %[[X_VAL:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFimplicit_dsa_taskloop_test1Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      ! CHECK: %[[LOAD_Z:.*]] = fir.load %[[DECL_Z]]#0 : !fir.ref<i32>
      x = y + z
      ! CHECK: hlfir.assign %{{.*}} to %[[X_VAL]]#0 : i32, !fir.ref<i32>
   end do
   !$omp end taskloop

   ! CHECK: omp.taskloop private(@[[TASKLOOP_TEST1_I_PRIVATE]] %[[DECL_I]]#0 -> %[[ARG0:.*]] : !fir.ref<i32>) {
   !$omp taskloop default(shared)
   do i = 1, 100
      ! CHECK:  %[[LOAD_Y:.*]] = fir.load %[[DECL_Y]]#0 : !fir.ref<i32>
      ! CHECK: %[[LOAD_Z:.*]] = fir.load %[[DECL_Z]]#0 : !fir.ref<i32>
      ! CHECK: %[[ADD_VAL:.*]] = arith.addi %[[LOAD_Y]], %[[LOAD_Z]] : i32
      x = y + z
      ! CHECK: hlfir.assign %[[ADD_VAL]] to %[[DECL_X]]#0 : i32, !fir.ref<i32>
   end do
   !$omp end taskloop
end subroutine

! Nested taskloop with implicit shared DSA variables.
! CHECK-LABEL: func @_QPimplicit_dsa_taskloop_test2
! CHECK:          %[[I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFimplicit_dsa_taskloop_test2Ei"}
! CHECK:          %[[I_DECL:.*]]:2 = hlfir.declare %[[I]] {uniq_name = "_QFimplicit_dsa_taskloop_test2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_taskloop_test2Ex"}
! CHECK:          %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_taskloop_test2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine implicit_dsa_taskloop_test2
   integer :: x
   ! CHECK:   omp.parallel {
   !$omp parallel 
   ! CHECK:   omp.taskloop private(@[[TASKLOOP_TEST2_I_PRIVATE]] %[[I_DECL]]#0 -> %[[ARG0:.*]] : !fir.ref<i32>) {
   !$omp taskloop
   do i = 1, 100
      ! CHECK: hlfir.assign %{{.*}} to %[[X_DECL]]#0 : i32, !fir.ref<i32>
      x = 2
   end do
   !$omp end taskloop

   ! CHECK: omp.taskloop private(@[[TASKLOOP_TEST2_X_PRIVATE]] %[[X_DECL]]#0 -> %[[ARG0]], @[[TASKLOOP_TEST2_I_PRIVATE]] %[[I_DECL]]#0 -> %[[ARG1:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
   !$omp taskloop private(x)
   do i = 1, 10
      ! CHECK: %[[DECL_PRIV_X:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFimplicit_dsa_taskloop_test2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      ! CHECK: %[[LOAD_X:.*]] = fir.load %[[DECL_PRIV_X]]#0 : !fir.ref<i32>
      x = x + 1
      ! CHECK: hlfir.assign %{{.*}} to %[[DECL_PRIV_X]]#0 : i32, !fir.ref<i32>
   end do
   !$omp end parallel

end subroutine

! Taskloop with implicit firstprivate DSA variables, enclosed in private context.

! CHECK-LABEL: func @_QPimplicit_dsa_taskloop_test3
! CHECK:          %[[I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFimplicit_dsa_taskloop_test3Ei"}
! CHECK:          %[[I_DECL:.*]]:2 = hlfir.declare %[[I]] {uniq_name = "_QFimplicit_dsa_taskloop_test3Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFimplicit_dsa_taskloop_test3Ex"}
! CHECK:          %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFimplicit_dsa_taskloop_test3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFimplicit_dsa_taskloop_test3Ey"}
! CHECK:          %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFimplicit_dsa_taskloop_test3Ey"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[Z:.*]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFimplicit_dsa_taskloop_test3Ez"}
! CHECK:          %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFimplicit_dsa_taskloop_test3Ez"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

subroutine implicit_dsa_taskloop_test3
   integer :: x, y, z
   ! CHECK:  omp.parallel private(@[[TASKLOOP_TEST3_X_FIRSTPRIVATE]] %[[X_DECL]]#0 -> %[[ARG0:.*]] : !fir.ref<i32>) {
   ! CHECK:  %[[X_PRIV_VAL:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFimplicit_dsa_taskloop_test3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
   !$omp parallel firstprivate(x)
   ! CHECK:  omp.taskloop private(@[[TASKLOOP_TEST3_X_FIRSTPRIVATE]] %[[X_PRIV_VAL]]#0 -> %[[ARG1:.*]], @[[TASKLOOP_TEST3_I_PRIVATE]] %[[I_DECL]]#0 -> %[[ARG2:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
   !$omp taskloop
   ! CHECK:  %[[X_VAL:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFimplicit_dsa_taskloop_test3Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
   do i = 1, 100
      ! CHECK: %[[LOAD_Y:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
      ! CHECK: %[[LOAD_Z:.*]] = fir.load %[[Z_DECL]]#0 : !fir.ref<i32>
      x = y + z
      ! CHECK: hlfir.assign %{{.*}} to %[[X_VAL]]#0 : i32, !fir.ref<i32>
   end do
   !$omp end taskloop
   !$omp end parallel
end subroutine


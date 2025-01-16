! REQUIRES: openmp_runtime

!RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPomp_task_simple() {
subroutine omp_task_simple
  !CHECK: omp.task {
  !$omp task
  !CHECK: fir.call @_QPfoo() {{.*}}: () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_simple

!===============================================================================
! `if` clause
!===============================================================================

!CHECK-LABEL: func @_QPomp_task_if(%{{.+}}) {
subroutine omp_task_if(bar)
  logical, intent(inout) :: bar
  !CHECK: omp.task if(%{{.+}}) {
  !$omp task if(bar)
  !CHECK: fir.call @_QPfoo() {{.*}}: () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_if

!===============================================================================
! `final` clause
!===============================================================================

!CHECK-LABEL: func @_QPomp_task_final(%{{.+}}) {
subroutine omp_task_final(bar)
  logical, intent(inout) :: bar
  !CHECK: omp.task final(%{{.+}}) {
  !$omp task final(bar)
  !CHECK: fir.call @_QPfoo() {{.*}}: () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_final

!===============================================================================
! `priority` clause
!===============================================================================

!CHECK-LABEL: func @_QPomp_task_priority(%{{.+}}) {
subroutine omp_task_priority(bar)
  integer, intent(inout) :: bar
  !CHECK: omp.task priority(%{{.+}}) {
  !$omp task priority(bar)
  !CHECK: fir.call @_QPfoo() {{.*}}: () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_priority

!===============================================================================
! `allocate` clause
!===============================================================================

!CHECK-LABEL: func @_QPtask_allocate
subroutine task_allocate()
  use omp_lib
  integer :: x
  !CHECK: omp.task allocate(%{{.+}} : i64 -> %{{.+}} : !fir.ref<i32>) {
  !$omp task allocate(omp_high_bw_mem_alloc: x) private(x)
  !CHECK: arith.addi
  x = x + 12
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_allocate

!===============================================================================
! `depend` clause
!===============================================================================

!CHECK-LABEL: func @_QPtask_depend
subroutine task_depend()
  integer :: x
  !CHECK: omp.task depend(taskdependin -> %{{.+}} : !fir.ref<i32>) {
  !$omp task depend(in : x)
  !CHECK: arith.addi
  x = x + 12
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_depend

!CHECK-LABEL: func @_QPtask_depend_non_int
subroutine task_depend_non_int()
  character(len = 15) :: x
  integer, allocatable :: y
  complex :: z
  !CHECK: omp.task depend(taskdependin -> %{{.+}} : !fir.ref<!fir.char<1,15>>, taskdependin -> %{{.+}} : !fir.ref<!fir.box<!fir.heap<i32>>>, taskdependin ->  %{{.+}} : !fir.ref<complex<f32>>) {
  !$omp task depend(in : x, y, z)
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_depend_non_int

!CHECK-LABEL: func @_QPtask_depend_all_kinds_one_task
subroutine task_depend_all_kinds_one_task()
  integer :: x
  !CHECK: omp.task depend(taskdependin -> %{{.+}} : !fir.ref<i32>, taskdependout -> %{{.+}} : !fir.ref<i32>, taskdependinout -> %{{.+}} : !fir.ref<i32>) {
  !$omp task depend(in : x) depend(out : x) depend(inout : x)
  !CHECK: arith.addi
  x = x + 12
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_depend_all_kinds_one_task

!CHECK-LABEL: func @_QPtask_depend_multi_var
subroutine task_depend_multi_var()
  integer :: x
  integer :: y
  !CHECK: omp.task depend(taskdependin -> %{{.*}} : !fir.ref<i32>, taskdependin -> %{{.+}} : !fir.ref<i32>) private({{.*x_firstprivate.*}}, {{.*y_firstprivate.*}}) {
  !$omp task depend(in :x,y)
  !CHECK: arith.addi
  x = x + 12
  y = y + 12
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_depend_multi_var

!CHECK-LABEL: func @_QPtask_depend_multi_task
subroutine task_depend_multi_task()
  integer :: x
  !CHECK: omp.task depend(taskdependout -> %{{.+}} : !fir.ref<i32>)
  !$omp task depend(out : x)
  !CHECK: arith.addi
  x = x + 12
  !CHECK: omp.terminator
  !$omp end task
  !CHECK: omp.task depend(taskdependinout -> %{{.+}} : !fir.ref<i32>)
  !$omp task depend(inout : x)
  !CHECK: arith.addi
  x = x + 12
  !CHECK: omp.terminator
  !$omp end task
  !CHECK: omp.task depend(taskdependin -> %{{.+}} : !fir.ref<i32>)
  !$omp task depend(in : x)
  !CHECK: arith.addi
  x = x + 12
  !CHECK: omp.terminator
  !$omp end task
  !CHECK: omp.task depend(taskdependmutexinoutset -> %{{.+}} : !fir.ref<i32>)
  !$omp task depend(mutexinoutset : x)
  !CHECK: arith.subi
  x = x - 12
  !CHECK: omp.terminator
  !$omp end task
    !CHECK: omp.task depend(taskdependinoutset -> %{{.+}} : !fir.ref<i32>)
  !$omp task depend(inoutset : x)
  !CHECK: arith.subi
  x = x - 12
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_depend_multi_task

!===============================================================================
! `private` clause
!===============================================================================
!CHECK-LABEL: func @_QPtask_private
subroutine task_private
  type mytype
  integer :: x
  end type mytype

!CHECK: %[[INT_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "int_var", uniq_name = "_QFtask_privateEint_var"}
!CHECK: %[[INT_VAR:.+]]:2 = hlfir.declare %[[INT_ALLOCA]] {uniq_name = "_QFtask_privateEint_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[MYTYPE_ALLOCA:.*]] = fir.alloca !fir.type<_QFtask_privateTmytype{x:i32}> {bindc_name = "mytype_var", uniq_name = "_QFtask_privateEmytype_var"}
!CHECK: %[[MYTYPE_VAR:.+]]:2 = hlfir.declare %[[MYTYPE_ALLOCA]] {uniq_name = "_QFtask_privateEmytype_var"} : (!fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>) -> (!fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>, !fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>)
  integer :: int_var
  type(mytype) :: mytype_var

  !CHECK: fir.call @_QPbar(%[[INT_VAR]]#1, %[[MYTYPE_VAR]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>) -> ()
  call bar(int_var, mytype_var)

  !CHECK: omp.task private(@{{.*int_var_private.*}} %[[INT_VAR]]#0 -> %[[INT_VAR_ARG:.*]], @{{.*mytype_var_private.*}} %[[MYTYPE_VAR]]#0 -> %[[MYTYPE_VAR_ARG:.*]] : !fir.ref<i32>, !fir.ref<!fir.type<{{.*}}>) {
  !$omp task private(int_var, mytype_var)
!CHECK: %[[INT_VAR_PRIVATE:.+]]:2 = hlfir.declare %[[INT_VAR_ARG]] {uniq_name = "_QFtask_privateEint_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[MYTYPE_VAR_PRIVATE:.+]]:2 = hlfir.declare %[[MYTYPE_VAR_ARG]] {uniq_name = "_QFtask_privateEmytype_var"} : (!fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>) -> (!fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>, !fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>)
!CHECK: fir.call @_QPbar(%[[INT_VAR_PRIVATE]]#1, %[[MYTYPE_VAR_PRIVATE]]#1) fastmath<contract> : (!fir.ref<i32>, !fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>) -> ()
  call bar(int_var, mytype_var)
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_private

!===============================================================================
! `firstprivate` clause
!===============================================================================
!CHECK-LABEL: func @_QPtask_firstprivate
subroutine task_firstprivate
  type mytype
  integer :: x
  end type mytype
 
  !CHECK: %[[INT_ALLOCA:.+]] = fir.alloca i32 {bindc_name = "int_var", uniq_name = "_QFtask_firstprivateEint_var"}
  !CHECK: %[[INT_VAR:.+]]:2 = hlfir.declare %[[INT_ALLOCA]] {uniq_name = "_QFtask_firstprivateEint_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !CHECK: %[[MYTYPE_ALLOCA:.+]] = fir.alloca !fir.type<_QFtask_firstprivateTmytype{x:i32}> {bindc_name = "mytype_var", uniq_name = "_QFtask_firstprivateEmytype_var"}
  !CHECK: %[[MYTYPE_VAR:.+]]:2 = hlfir.declare %[[MYTYPE_ALLOCA]] {uniq_name = "_QFtask_firstprivateEmytype_var"} : (!fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>) -> (!fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>, !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>)
  integer :: int_var
  type(mytype) :: mytype_var

!CHECK: fir.call @_QPbaz(%[[INT_VAR]]#1, %[[MYTYPE_VAR]]#1) fastmath<contract> : (!fir.ref<i32>, !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>) -> ()
  call baz(int_var, mytype_var)

  !CHECK: omp.task private(@{{.*int_var_firstprivate.*}} %[[INT_VAR]]#0 -> %[[INT_VAR_ARG:.*]], @{{.*mytype_var_firstprivate.*}} %[[MYTYPE_VAR]]#0 -> %[[MYTYPE_VAR_ARG:.*]] : !fir.ref<i32>, !fir.ref<{{.*}}) {
  !$omp task firstprivate(int_var, mytype_var)
!CHECK: %[[INT_VAR_FIRSTPRIVATE:.+]]:2 = hlfir.declare %[[INT_VAR_ARG]] {uniq_name = "_QFtask_firstprivateEint_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[MYTYPE_VAR_FIRSTPRIVATE:.+]]:2 = hlfir.declare %[[MYTYPE_VAR_ARG]] {uniq_name = "_QFtask_firstprivateEmytype_var"} : (!fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>) -> (!fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>, !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>)
  call baz(int_var, mytype_var)
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_firstprivate

!===============================================================================
! Multiple clauses
!===============================================================================

!CHECK-LABEL: func @_QPtask_multiple_clauses
subroutine task_multiple_clauses()
  use omp_lib

!CHECK: %[[X_ALLOCA:.+]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtask_multiple_clausesEx"}
!CHECK: %[[X:.+]]:2 = hlfir.declare %[[X_ALLOCA]] {uniq_name = "_QFtask_multiple_clausesEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y_ALLOCA:.+]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFtask_multiple_clausesEy"}
!CHECK: %[[Y:.+]]:2 = hlfir.declare %[[Y_ALLOCA]] {uniq_name = "_QFtask_multiple_clausesEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Z_ALLOCA:.+]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFtask_multiple_clausesEz"}
!CHECK: %[[Z:.+]]:2 = hlfir.declare %[[Z_ALLOCA]] {uniq_name = "_QFtask_multiple_clausesEz"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer :: x, y, z
  logical :: buzz

  !CHECK: omp.task allocate(%{{.+}} : i64 -> %{{.+}} : !fir.ref<i32>) final(%{{.+}}) if(%{{.+}}) priority(%{{.+}}) private({{.*}}) {
  !$omp task if(buzz) final(buzz) priority(z) allocate(omp_high_bw_mem_alloc: x) private(x) firstprivate(y)

!CHECK: %[[X_PRIV:.+]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtask_multiple_clausesEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y_PRIV:.+]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFtask_multiple_clausesEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  !CHECK: arith.addi
  x = x + 12
  !CHECK: arith.subi
  y = y - 12

  !CHECK: omp.terminator
  !$omp end task
end subroutine task_multiple_clauses

subroutine task_mergeable()
!CHECK: omp.task mergeable {
!CHECK: omp.terminator
!CHECK: }
 !$omp task mergeable
 !$omp end task
end subroutine

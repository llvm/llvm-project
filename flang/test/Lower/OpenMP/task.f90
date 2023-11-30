!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

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
  !CHECK: omp.task allocate(%{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>) {
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
  !CHECK: omp.task depend(taskdependin -> %{{.+}} : !fir.ref<!fir.char<1,15>>, taskdependin -> %{{.+}} : !fir.ref<!fir.box<!fir.heap<i32>>>, taskdependin ->  %{{.+}} : !fir.ref<!fir.complex<4>>) {
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
  !CHECK: omp.task depend(taskdependin -> %{{.*}} : !fir.ref<i32>, taskdependin -> %{{.+}} : !fir.ref<i32>) {
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

  !CHECK: omp.task {
  !$omp task private(int_var, mytype_var)
!CHECK: %[[INT_PRIVATE_ALLOCA:.+]] = fir.alloca i32 {bindc_name = "int_var", pinned, uniq_name = "_QFtask_privateEint_var"}
!CHECK: %[[INT_VAR_PRIVATE:.+]]:2 = hlfir.declare %[[INT_PRIVATE_ALLOCA]] {uniq_name = "_QFtask_privateEint_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[MYTYPE_PRIVATE_ALLOCA:.+]] = fir.alloca !fir.type<_QFtask_privateTmytype{x:i32}> {bindc_name = "mytype_var", pinned, uniq_name = "_QFtask_privateEmytype_var"}
!CHECK: %[[MYTYPE_VAR_PRIVATE:.+]]:2 = hlfir.declare %[[MYTYPE_PRIVATE_ALLOCA]] {uniq_name = "_QFtask_privateEmytype_var"} : (!fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>) -> (!fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>, !fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>)
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

  !CHECK: omp.task {
  !$omp task firstprivate(int_var, mytype_var)
!CHECK: %[[INT_FIRSTPRIVATE_ALLOCA:.+]] = fir.alloca i32 {bindc_name = "int_var", pinned, uniq_name = "_QFtask_firstprivateEint_var"}
!CHECK: %[[INT_VAR_FIRSTPRIVATE:.+]]:2 = hlfir.declare %[[INT_FIRSTPRIVATE_ALLOCA]] {uniq_name = "_QFtask_firstprivateEint_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[INT_VAR_LOAD:.+]] = fir.load %[[INT_VAR]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[INT_VAR_LOAD]] to %[[INT_VAR_FIRSTPRIVATE]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK: %[[MYTYPE_FIRSTPRIVATE_ALLOCA:.+]] = fir.alloca !fir.type<_QFtask_firstprivateTmytype{x:i32}> {bindc_name = "mytype_var", pinned, uniq_name = "_QFtask_firstprivateEmytype_var"}
!CHECK: %[[MYTYPE_VAR_FIRSTPRIVATE:.+]]:2 = hlfir.declare %[[MYTYPE_FIRSTPRIVATE_ALLOCA]] {uniq_name = "_QFtask_firstprivateEmytype_var"} : (!fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>) -> (!fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>, !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>)
!CHECK: hlfir.assign %[[MYTYPE_VAR]]#0 to %[[MYTYPE_VAR_FIRSTPRIVATE]]#0 temporary_lhs : !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>, !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>
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

  !CHECK: omp.task if(%{{.+}}) final(%{{.+}}) priority(%{{.+}}) allocate(%{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>) {
  !$omp task if(buzz) final(buzz) priority(z) allocate(omp_high_bw_mem_alloc: x) private(x) firstprivate(y)

!CHECK: %[[X_PRIV_ALLOCA:.+]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFtask_multiple_clausesEx"}
!CHECK: %[[X_PRIV:.+]]:2 = hlfir.declare %[[X_PRIV_ALLOCA]] {uniq_name = "_QFtask_multiple_clausesEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y_PRIV_ALLOCA:.+]] = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFtask_multiple_clausesEy"}
!CHECK: %[[Y_PRIV:.+]]:2 = hlfir.declare %[[Y_PRIV_ALLOCA]] {uniq_name = "_QFtask_multiple_clausesEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y_LOAD:.+]] = fir.load %[[Y]]#0 : !fir.ref<i32>
!CHECK: hlfir.assign %[[Y_LOAD]] to %[[Y_PRIV]]#0 temporary_lhs : i32, !fir.ref<i32>

  !CHECK: arith.addi
  x = x + 12
  !CHECK: arith.subi
  y = y - 12

  !CHECK: omp.terminator
  !$omp end task
end subroutine task_multiple_clauses

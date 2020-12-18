! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test lowering of allocatables using runtime for allocate/deallcoate statements.
! CHECK-LABEL: _QPfooscalar
subroutine fooscalar()
  ! Test lowering of local allocatable specification
  real, allocatable :: x
  ! CHECK: %[[xAddrVar:.*]] = fir.alloca !fir.heap<f32> {name = "_QFfooscalarEx.addr"}
  ! CHECK: %[[nullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<f32>
  ! CHECK: fir.store %[[nullAddr]] to %[[xAddrVar]] : !fir.ref<!fir.heap<f32>>

  ! Test allocation of local allocatables
  allocate(x)
  ! CHECK: %[[alloc:.*]] = fir.allocmem f32 {name = "_QFfooscalarEx.alloc"}
  ! CHECK: fir.store %[[alloc]] to %[[xAddrVar]] : !fir.ref<!fir.heap<f32>>

  ! Test reading allocatable bounds and extents
  print *, x
  ! CHECK: %[[xAddr1:.*]] = fir.load %[[xAddrVar]] : !fir.ref<!fir.heap<f32>>
  ! CHECK: = fir.load %[[xAddr1]] : !fir.heap<f32>

  ! Test deallocation
  deallocate(x)
  ! CHECK: %[[xAddr2:.*]] = fir.load %[[xAddrVar]] : !fir.ref<!fir.heap<f32>>
  ! CHECK: fir.freemem %[[xAddr2]] : !fir.heap<f32>
  ! CHECK: %[[nullAddr1:.*]] = fir.convert %c0_0 : (index) -> !fir.heap<f32>
  ! fir.store %[[nullAddr1]] to %[[xAddrVar]] : !fir.ref<!fir.heap<f32>>
end subroutine

! CHECK-LABEL: _QPfoodim1
subroutine foodim1()
  ! Test lowering of local allocatable specification
  real, allocatable :: x(:)
  ! CHECK-DAG: %[[xAddrVar:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>> {name = "_QFfoodim1Ex.addr"}
  ! CHECK-DAG: %[[xLbVar:.*]] = fir.alloca index {name = "_QFfoodim1Ex.lb0"}
  ! CHECK-DAG: %[[xExtVar:.*]] = fir.alloca index {name = "_QFfoodim1Ex.ext0"}
  ! CHECK: %[[nullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: fir.store %[[nullAddr]] to %[[xAddrVar]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>

  ! Test allocation of local allocatables
  allocate(x(42:100))
  ! CHECK-DAG: %[[c42:.*]] = fir.convert %c42{{.*}} : (i32) -> index
  ! CHECK-DAG: %[[c100:.*]] = fir.convert %c100_i32 : (i32) -> index
  ! CHECK-DAG: %[[diff:.*]] = subi %[[c100]], %[[c42]] : index
  ! CHECK: %[[extent:.*]] = addi %[[diff]], %c1{{.*}} : index
  ! CHECK: %[[alloc:.*]] = fir.allocmem !fir.array<?xf32>, %[[extent]] {name = "_QFfoodim1Ex.alloc"}
  ! CHECK-DAG: fir.store %[[alloc]] to %[[xAddrVar]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  ! CHECK-DAG: fir.store %[[extent]] to %[[xExtVar]] : !fir.ref<index>
  ! CHECK-DAG: fir.store %[[c42]] to %[[xLbVar]] : !fir.ref<index>

  ! Test reading allocatable bounds and extents
  print *, x(42)
  ! CHECK-DAG: fir.load %[[xExtVar]] : !fir.ref<index>
  ! CHECK-DAG: fir.load %[[xLbVar]] : !fir.ref<index>
  ! CHECK-DAG: fir.load %[[xAddrVar]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>

  deallocate(x)
  ! CHECK: %[[xAddr1:.*]] = fir.load %1 : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: fir.freemem %[[xAddr1]] : !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[nullAddr1:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: fir.store %[[nullAddr1]] to %[[xAddrVar]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
end subroutine

! CHECK-LABEL: _QPfoodim2
subroutine foodim2()
  ! Test lowering of local allocatable specification
  real, allocatable :: x(:, :)
  ! CHECK-DAG: fir.alloca !fir.heap<!fir.array<?x?xf32>> {name = "_QFfoodim2Ex.addr"}
  ! CHECK-DAG: fir.alloca index {name = "_QFfoodim2Ex.lb0"}
  ! CHECK-DAG: fir.alloca index {name = "_QFfoodim2Ex.ext0"}
  ! CHECK-DAG: fir.alloca index {name = "_QFfoodim2Ex.lb1"}
  ! CHECK-DAG: fir.alloca index {name = "_QFfoodim2Ex.ext1"}
end subroutine

! test lowering of character allocatables. Focus is placed on the length handling
! CHECK-LABEL: _QPchar_deferred(
subroutine char_deferred(n)
  integer :: n
  character(:), allocatable :: c
  ! CHECK-DAG: %[[cAddrVar:.*]] = fir.alloca !fir.heap<!fir.char<1,?>> {name = "_QFchar_deferredEc.addr"}
  ! CHECK-DAG: %[[cLenVar:.*]] = fir.alloca index {name = "_QFchar_deferredEc.len"}
  allocate(character(10):: c)
  ! CHECK: %[[c10:.]] = fir.convert %c10_i32 : (i32) -> index
  ! CHECK: fir.allocmem !fir.char<1,?>, %[[c10]] {name = "_QFchar_deferredEc.alloc"}
  ! CHECK: fir.store %[[c10]] to %[[cLenVar]] : !fir.ref<index>
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}} : !fir.heap<!fir.char<1,?>>
  allocate(character(n):: c)
  ! CHECK: %[[n:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK: %[[ni:.*]] = fir.convert %[[n]] : (i32) -> index
  ! CHECK: fir.allocmem !fir.char<1,?>, %[[ni]] {name = "_QFchar_deferredEc.alloc"}
  ! CHECK: fir.store %[[ni]] to %[[cLenVar]] : !fir.ref<index>

  call bar(c)
  ! CHECK-DAG: %[[cLen:.*]] = fir.load %[[cLenVar]] : !fir.ref<index>
  ! CHECK-DAG: %[[cAddr:.*]] = fir.load %[[cAddrVar]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
  ! CHECK-DAG: %[[cAddrcast:.*]] = fir.convert %[[cAddr]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: fir.emboxchar %[[cAddrcast]], %[[cLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
end subroutine

! CHECK-LABEL: _QPchar_explicit_cst(
subroutine char_explicit_cst(n)
  integer :: n
  character(10), allocatable :: c
  ! CHECK-DAG: %[[cLen:.*]] = constant 10 : index
  ! CHECK-DAG: %[[cAddrVar:.*]] = fir.alloca !fir.heap<!fir.char<1,10>> {name = "_QFchar_explicit_cstEc.addr"}
  ! CHECK-NOT: "_QFchar_explicit_cstEc.len"
  allocate(c)
  ! CHECK: fir.allocmem !fir.char<1,10> {name = "_QFchar_explicit_cstEc.alloc"}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}} : !fir.heap<!fir.char<1,10>>
  allocate(character(n):: c)
  ! CHECK: fir.allocmem !fir.char<1,10> {name = "_QFchar_explicit_cstEc.alloc"}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}} : !fir.heap<!fir.char<1,10>>
  allocate(character(10):: c)
  ! CHECK: fir.allocmem !fir.char<1,10> {name = "_QFchar_explicit_cstEc.alloc"}
  call bar(c)
  ! CHECK: %[[cAddr:.*]] = fir.load %[[cAddrVar]] : !fir.ref<!fir.heap<!fir.char<1,10>>>
  ! CHECK: %[[cAddrcast:.*]] = fir.convert %[[cAddr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: fir.emboxchar %[[cAddrcast]], %[[cLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
end subroutine

! CHECK-LABEL: _QPchar_explicit_dyn(
subroutine char_explicit_dyn(l1, l2)
  integer :: l1, l2
  character(l1), allocatable :: c
  ! CHECK-DAG: %[[cLen:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[cAddrVar:.*]] = fir.alloca !fir.heap<!fir.char<1,?>> {name = "_QFchar_explicit_dynEc.addr"}
  ! CHECK-NOT: "_QFchar_explicit_dynEc.len"
  allocate(c)
  ! CHECK: %[[cLenCast1:.*]] = fir.convert %[[cLen]] : (i32) -> index
  ! CHECK: fir.allocmem !fir.char<1,?>, %[[cLenCast1]] {name = "_QFchar_explicit_dynEc.alloc"}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}} : !fir.heap<!fir.char<1,?>>
  allocate(character(l2):: c)
  ! CHECK: %[[cLenCast2:.*]] = fir.convert %[[cLen]] : (i32) -> index
  ! CHECK: fir.allocmem !fir.char<1,?>, %[[cLenCast2]] {name = "_QFchar_explicit_dynEc.alloc"}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}} : !fir.heap<!fir.char<1,?>>
  allocate(character(10):: c)
  ! CHECK: %[[cLenCast3:.*]] = fir.convert %[[cLen]] : (i32) -> index
  ! CHECK: fir.allocmem !fir.char<1,?>, %[[cLenCast3]] {name = "_QFchar_explicit_dynEc.alloc"}
  call bar(c)
  ! CHECK-DAG: %[[cLenCast4:.*]] = fir.convert %[[cLen]] : (i32) -> index
  ! CHECK-DAG: %[[cAddr:.*]] = fir.load %[[cAddrVar]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
  ! CHECK-DAG: %[[cAddrcast:.*]] = fir.convert %[[cAddr]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: fir.emboxchar %[[cAddrcast]], %[[cLenCast4]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
end subroutine

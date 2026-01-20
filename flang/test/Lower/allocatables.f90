! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Test lowering of allocatables using runtime for allocate/deallcoate statements.
! CHECK-LABEL: _QPfooscalar
subroutine fooscalar()
  ! Test lowering of local allocatable specification
  real, allocatable :: x
  ! CHECK: %[[xAddrVar:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {{{.*}}uniq_name = "_QFfooscalarEx"}
  ! CHECK: %[[nullAddr:.*]] = fir.zero_bits !fir.heap<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[nullAddr]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  ! CHECK: fir.store %[[box]] to %[[xAddrVar]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[decl:.*]]:2 = hlfir.declare %[[xAddrVar]] {{{.*}}uniq_name = "_QFfooscalarEx"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  ! Test allocation of local allocatables
  allocate(x)
  ! CHECK: %[[alloc:.*]] = fir.allocmem f32 {{{.*}}uniq_name = "_QFfooscalarEx.alloc"}
  ! CHECK: %[[box2:.*]] = fir.embox %[[alloc]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  ! CHECK: fir.store %[[box2]] to %[[decl]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>

  ! Test reading allocatable bounds and extents
  print *, x
  ! CHECK: %[[xAddr1:.*]] = fir.load %[[decl]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[xAddr1]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: = fir.load %[[addr]] : !fir.heap<f32>

  ! Test deallocation
  deallocate(x)
  ! CHECK: %[[xAddr2:.*]] = fir.load %[[decl]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr2:.*]] = fir.box_addr %[[xAddr2]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: fir.freemem %[[addr2]]
  ! CHECK: %[[nullAddr1:.*]] = fir.zero_bits !fir.heap<f32>
  ! CHECK: %[[box3:.*]] = fir.embox %[[nullAddr1]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  ! fir.store %[[box3]] to %[[decl]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
end subroutine

! CHECK-LABEL: _QPfoodim1
subroutine foodim1()
  ! Test lowering of local allocatable specification
  real, allocatable :: x(:)
  ! CHECK: %[[xAddrVar:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {{{.*}}uniq_name = "_QFfoodim1Ex"}
  ! CHECK: %[[nullAddr:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[box:.*]] = fir.embox %[[nullAddr]]
  ! CHECK: fir.store %[[box]] to %[[xAddrVar]]

  ! Test allocation of local allocatables
  allocate(x(42:100))
  ! CHECK: %[[c42:.*]] = fir.convert %c42{{.*}} : (i32) -> index
  ! CHECK: %[[c100:.*]] = fir.convert %c100_i32 : (i32) -> index
  ! CHECK: %[[diff:.*]] = arith.subi %[[c100]], %[[c42]] : index
  ! CHECK: %[[rawExtent:.*]] = arith.addi %[[diff]], %c1{{.*}} : index
  ! CHECK: %[[extentPositive:.*]] = arith.cmpi sgt, %[[rawExtent]], %c0{{.*}} : index
  ! CHECK: %[[extent:.*]] = arith.select %[[extentPositive]], %[[rawExtent]], %c0{{.*}} : index
  ! CHECK: %[[alloc:.*]] = fir.allocmem !fir.array<?xf32>, %[[extent]] {{{.*}}uniq_name = "_QFfoodim1Ex.alloc"}
  ! CHECK: %[[shape:.*]] = fir.shape_shift %[[c42]], %[[extent]]
  ! CHECK: %[[box2:.*]] = fir.embox %[[alloc]](%[[shape]])
  ! CHECK: fir.store %[[box2]] to %{{.*}}

  ! Test reading allocatable bounds and extents
  print *, x(42)
  ! CHECK: %[[load:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: hlfir.designate %[[load]] (%c42)

  deallocate(x)
  ! CHECK: %[[load2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load2]]
  ! CHECK: fir.freemem %[[addr]]
end subroutine

! CHECK-LABEL: _QPfoodim2
subroutine foodim2()
  ! Test lowering of local allocatable specification
  real, allocatable :: x(:, :)
  ! CHECK: fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>> {{{.*}}uniq_name = "_QFfoodim2Ex"}
end subroutine

! test lowering of character allocatables. Focus is placed on the length handling
! CHECK-LABEL: _QPchar_deferred(
subroutine char_deferred(n)
  integer :: n
  character(:), allocatable :: c
  ! CHECK: %[[cAddrVar:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFchar_deferredEc"}
  allocate(character(10):: c)
  ! CHECK: %[[c10:.]] = fir.convert %c10_i32 : (i32) -> index
  ! CHECK: %[[alloc:.*]] = fir.allocmem !fir.char<1,?>(%[[c10]] : index) {{{.*}}uniq_name = "_QFchar_deferredEc.alloc"}
  ! CHECK: %[[box:.*]] = fir.embox %[[alloc]] typeparams %[[c10]]
  ! CHECK: fir.store %[[box]] to %{{.*}}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}}
  allocate(character(n):: c)
  ! CHECK: %[[n:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[nPositive:.*]] = arith.cmpi sgt, %[[n]], %c0{{.*}} : i32
  ! CHECK: %[[ns:.*]] = arith.select %[[nPositive]], %[[n]], %c0{{.*}} : i32
  ! CHECK: %[[ni:.*]] = fir.convert %[[ns]] : (i32) -> index
  ! CHECK: %[[alloc2:.*]] = fir.allocmem !fir.char<1,?>(%[[ni]] : index)
  ! CHECK: %[[box2:.*]] = fir.embox %[[alloc2]] typeparams %[[ni]]
  ! CHECK: fir.store %[[box2]] to %{{.*}}

  call bar(c)
  ! CHECK: %[[load:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]]
  ! CHECK: %[[load2:.*]] = fir.load %{{.*}}
  ! CHECK: %[[len:.*]] = fir.box_elesize %[[load2]]
  ! CHECK: fir.emboxchar %[[addr]], %[[len]]
end subroutine

! CHECK-LABEL: _QPchar_explicit_cst(
subroutine char_explicit_cst(n)
  integer :: n
  character(10), allocatable :: c
  ! CHECK: %[[cAddrVar:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,10>>> {{{.*}}uniq_name = "_QFchar_explicit_cstEc"}
  allocate(c)
  ! CHECK: fir.allocmem !fir.char<1,10> {{{.*}}uniq_name = "_QFchar_explicit_cstEc.alloc"}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}}
  allocate(character(n):: c)
  ! CHECK: fir.allocmem !fir.char<1,10> {{{.*}}uniq_name = "_QFchar_explicit_cstEc.alloc"}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}}
  allocate(character(10):: c)
  ! CHECK: fir.allocmem !fir.char<1,10> {{{.*}}uniq_name = "_QFchar_explicit_cstEc.alloc"}
  call bar(c)
  ! CHECK: %[[load:.*]] = fir.load %{{.*}}
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[addr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: fir.emboxchar %[[cast]], %c10
end subroutine

! CHECK-LABEL: _QPchar_explicit_dyn(
subroutine char_explicit_dyn(l1, l2)
  integer :: l1, l2
  character(l1), allocatable :: c
  ! CHECK: %[[cAddrVar:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {{{.*}}uniq_name = "_QFchar_explicit_dynEc"}
  ! CHECK: %[[l1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
  ! CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
  ! CHECK: %[[cmp:.*]] = arith.cmpi sgt, %[[l1]], %[[c0_i32]] : i32
  ! CHECK: %[[cLen:.*]] = arith.select %[[cmp]], %[[l1]], %[[c0_i32]] : i32
  allocate(c)
  ! CHECK: %[[cLenCast1:.*]] = fir.convert %[[cLen]] : (i32) -> index
  ! CHECK: fir.allocmem !fir.char<1,?>(%[[cLenCast1]] : index) {{{.*}}uniq_name = "_QFchar_explicit_dynEc.alloc"}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}}
  allocate(character(l2):: c)
  ! CHECK: %[[cLenCast2:.*]] = fir.convert %[[cLen]] : (i32) -> index
  ! CHECK: fir.allocmem !fir.char<1,?>(%[[cLenCast2]] : index) {{{.*}}uniq_name = "_QFchar_explicit_dynEc.alloc"}
  deallocate(c)
  ! CHECK: fir.freemem %{{.*}}
  allocate(character(10):: c)
  ! CHECK: %[[cLenCast3:.*]] = fir.convert %[[cLen]] : (i32) -> index
  ! CHECK: fir.allocmem !fir.char<1,?>(%[[cLenCast3]] : index) {{{.*}}uniq_name = "_QFchar_explicit_dynEc.alloc"}
  call bar(c)
  ! CHECK: %[[load:.*]] = fir.load %{{.*}}
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[load]]
  ! CHECK: fir.emboxchar %[[addr]], %[[cLen]]
end subroutine

! CHECK-LABEL: _QPspecifiers(
subroutine specifiers
  allocatable jj1(:), jj2(:,:), jj3(:)
  ! CHECK: [[STAT:%[0-9]+]] = fir.alloca i32 {{{.*}}uniq_name = "_QFspecifiersEsss"}
  integer sss
  character*30 :: mmm = "None"
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.store [[RESULT]] to %{{.*}}
  ! CHECK: fir.if %{{[0-9]+}} {
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.store [[RESULT]] to %{{.*}}
  ! CHECK: fir.if %{{[0-9]+}} {
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.store [[RESULT]] to %{{.*}}
  ! CHECK-NOT: fir.if %{{[0-9]+}} {
  ! CHECK-COUNT-2: }
  ! CHECK-NOT: }
  allocate(jj1(3), jj2(3,3), jj3(3), stat=sss, errmsg=mmm)
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  ! CHECK: fir.call @_FortranAAllocatableSetBounds
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableAllocate
  allocate(jj1(3), jj2(3,3), jj3(3), stat=sss, errmsg=mmm)
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: fir.store [[RESULT]] to %{{.*}}
  ! CHECK: fir.if %{{[0-9]+}} {
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: fir.store [[RESULT]] to %{{.*}}
  ! CHECK: fir.if %{{[0-9]+}} {
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: fir.store [[RESULT]] to %{{.*}}
  ! CHECK-NOT: fir.if %{{[0-9]+}} {
  ! CHECK-COUNT-2: }
  ! CHECK-NOT: }
  deallocate(jj1, jj2, jj3, stat=sss, errmsg=mmm)
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  ! CHECK: [[RESULT:%[0-9]+]] = fir.call @_FortranAAllocatableDeallocate
  deallocate(jj1, jj2, jj3, stat=sss, errmsg=mmm)
end subroutine specifiers

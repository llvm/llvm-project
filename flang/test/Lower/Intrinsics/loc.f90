! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! Test LOC intrinsic

! CHECK-LABEL: func.func @_QPloc_scalar() {
subroutine loc_scalar()
  integer(8) :: p
  integer :: x
  p = loc(x)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[x:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[xaddrval:.*]] = fir.convert %[[xaddr]] : (!fir.ref<i32>) -> i64
! CHECK: fir.store %[[xaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_char() {
subroutine loc_char()
  integer(8) :: p
  character(5) :: x = "abcde"
  p = loc(x)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[x:.*]] = fir.address_of(@_QFloc_charEx) : !fir.ref<!fir.char<1,5>>
! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ref<!fir.char<1,5>>) -> !fir.box<!fir.char<1,5>>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,5>>
! CHECK: %[[xaddrval:.*]] = fir.convert %[[xaddr]] : (!fir.ref<!fir.char<1,5>>) -> i64
! CHECK: fir.store %[[xaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_substring() {
subroutine loc_substring()
  integer(8) :: p
  character(5) :: x = "abcde"
  p = loc(x(2:))
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[x:.*]] = fir.address_of(@_QFloc_substringEx) : !fir.ref<!fir.char<1,5>>
! CHECK: %[[sslb:.*]] = arith.constant 2 : i64
! CHECK: %[[ssub:.*]] = arith.constant 5 : i64
! CHECK: %[[sslbidx:.*]] = fir.convert %[[sslb]] : (i64) -> index
! CHECK: %[[ssubidx:.*]] = fir.convert %[[ssub]] : (i64) -> index
! CHECK: %[[one:.*]] = arith.constant 1 : index
! CHECK: %[[lboffset:.*]] = arith.subi %[[sslbidx]], %c1 : index
! CHECK: %[[xarr:.*]] = fir.convert %[[x]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.array<5x!fir.char<1>>>
! CHECK: %[[xarrcoord:.*]] = fir.coordinate_of %[[xarr]], %[[lboffset]] : (!fir.ref<!fir.array<5x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK: %[[xss:.*]] = fir.convert %[[xarrcoord]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK: %[[rng:.*]] = arith.subi %[[ssubidx]], %[[sslbidx]] : index
! CHECK: %[[rngp1:.*]] = arith.addi %[[rng]], %[[one]] : index
! CHECK: %[[zero:.*]] = arith.constant 0 : index
! CHECK: %[[cmpval:.*]] = arith.cmpi slt, %[[rngp1]], %[[zero]] : index
! CHECK: %[[sltval:.*]] = arith.select %[[cmpval]], %[[zero]], %[[rngp1]] : index
! CHECK: %[[xssbox:.*]] = fir.embox %[[xss]] typeparams %[[sltval]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK: %[[xssaddr:.*]] = fir.box_addr %[[xssbox]] : (!fir.box<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
! CHECK: %[[xssaddrval:.*]] = fir.convert %[[xssaddr]] : (!fir.ref<!fir.char<1,?>>) -> i64
! CHECK: fir.store %[[xssaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_array() {
subroutine loc_array
  integer(8) :: p
  integer :: x(10)
  p = loc(x)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[ten:.*]] = arith.constant 10 : index
! CHECK: %[[x:.*]] = fir.alloca !fir.array<10xi32> {{.*}}
! CHECK: %[[xshp:.*]] = fir.shape %[[ten]] : (index) -> !fir.shape<1>
! CHECK: %[[xbox:.*]] = fir.embox %[[x]](%[[xshp]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! CHECK: %[[xaddrval:.*]] = fir.convert %[[xaddr]] : (!fir.ref<!fir.array<10xi32>>) -> i64
! CHECK: fir.store %[[xaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_chararray() {
subroutine loc_chararray()
  integer(8) :: p
  character(5) :: x(2)
  p = loc(x)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[two:.*]] = arith.constant 2 : index
! CHECK: %[[x:.*]] = fir.alloca !fir.array<2x!fir.char<1,5>> {{.*}}
! CHECK: %[[xshp:.*]] = fir.shape %[[two]] : (index) -> !fir.shape<1>
! CHECK: %[[xbox:.*]] = fir.embox %[[x]](%[[xshp]]) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<!fir.array<2x!fir.char<1,5>>>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
! CHECK: %[[xaddrval:.*]] = fir.convert %[[xaddr]] : (!fir.ref<!fir.array<2x!fir.char<1,5>>>) -> i64
! CHECK: fir.store %[[xaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_arrayelement() {
subroutine loc_arrayelement()
  integer(8) :: p
  integer :: x(10)
  p = loc(x(7))
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[x:.*]] = fir.alloca !fir.array<10xi32> {{.*}}
! CHECK: %[[idx:.*]] = arith.constant 7 : i64
! CHECK: %[[lb:.*]] = arith.constant 1 : i64
! CHECK: %[[offset:.*]] = arith.subi %[[idx]], %[[lb]] : i64
! CHECK: %[[xelemcoord:.*]] = fir.coordinate_of %[[x]], %[[offset]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[xelembox:.*]] = fir.embox %[[xelemcoord]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[xelemaddr:.*]] = fir.box_addr %[[xelembox]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK: %[[xelemaddrval:.*]] = fir.convert %[[xelemaddr]] : (!fir.ref<i32>) -> i64
! CHECK: fir.store %[[xelemaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_arraysection(
! CHECK-SAME: %[[arg:.*]]: !fir.ref<i32> {{.*}}) {
subroutine loc_arraysection(i)
  integer(8) :: p
  integer :: i
  real :: x(11)
  p = loc(x(i:))
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[eleven:.*]] = arith.constant 11 : index
! CHECK: %[[x:.*]] = fir.alloca !fir.array<11xf32> {{.*}}
! CHECK: %[[one:.*]] = arith.constant 1 : index
! CHECK: %[[i:.*]] = fir.load %[[arg]] : !fir.ref<i32>
! CHECK: %[[il:.*]] = fir.convert %[[i]] : (i32) -> i64
! CHECK: %[[iidx:.*]] = fir.convert %[[il]] : (i64) -> index
! CHECK: %[[onel:.*]] = arith.constant 1 : i64
! CHECK: %[[stpidx:.*]] = fir.convert %[[onel]] : (i64) -> index
! CHECK: %[[xrng:.*]] = arith.addi %[[one]], %[[eleven]] : index
! CHECK: %[[xub:.*]] = arith.subi %[[xrng]], %[[one]] : index
! CHECK: %[[xshp:.*]] = fir.shape %[[eleven]] : (index) -> !fir.shape<1>
! CHECK: %[[xslice:.*]] = fir.slice %[[iidx]], %[[xub]], %[[stpidx]] : (index, index, index) -> !fir.slice<1>
! CHECK: %[[xbox:.*]] = fir.embox %[[x]](%[[xshp]]) [%[[xslice]]] : (!fir.ref<!fir.array<11xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK: %[[xaddrval:.*]] = fir.convert %[[xaddr]] : (!fir.ref<!fir.array<?xf32>>) -> i64
! CHECK: fir.store %[[xaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_non_save_pointer_scalar() {
subroutine loc_non_save_pointer_scalar()
  integer(8) :: p
  real, pointer :: x
  real, target :: t
  x => t
  p = loc(x)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[t:.*]] = fir.alloca f32 {{.*}}
! CHECK: %2 = fir.alloca !fir.box<!fir.ptr<f32>> {{.*}}
! CHECK: %[[xa:.*]] = fir.alloca !fir.ptr<f32> {{.*}}
! CHECK: %[[zero:.*]] = fir.zero_bits !fir.ptr<f32>
! CHECK: fir.store %[[zero]] to %[[xa]] : !fir.ref<!fir.ptr<f32>>
! CHECK: %[[taddr:.*]] = fir.convert %[[t]] : (!fir.ref<f32>) -> !fir.ptr<f32>
! CHECK: fir.store %[[taddr]] to %[[xa]] : !fir.ref<!fir.ptr<f32>>
! CHECK: %[[x:.*]] = fir.load %[[xa]] : !fir.ref<!fir.ptr<f32>>
! CHECK: %[[xbox:.*]] = fir.embox %[[x]] : (!fir.ptr<f32>) -> !fir.box<f32>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<f32>) -> !fir.ref<f32>
! CHECK: %[[xaddrval:.*]] = fir.convert %[[xaddr]] : (!fir.ref<f32>) -> i64
! CHECK: fir.store %[[xaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_save_pointer_scalar() {
subroutine loc_save_pointer_scalar()
  integer :: p
  real, pointer, save :: x
  p = loc(x)
! CHECK: %[[p:.*]] = fir.alloca i32 {{.*}}
! CHECK: %[[x:.*]] = fir.address_of(@_QFloc_save_pointer_scalarEx) : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[xref:.*]] = fir.load %[[x]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xref]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK: %[[xbox:.*]] = fir.embox %[[xaddr]] : (!fir.ptr<f32>) -> !fir.box<f32>
! CHECK: %[[xaddr2:.*]] = fir.box_addr %[[xbox]] : (!fir.box<f32>) -> !fir.ref<f32>
! CHECK: %[[xaddr2vall:.*]] = fir.convert %[[xaddr2]] : (!fir.ref<f32>) -> i64
! CHECK: %[[xaddr2val:.*]] = fir.convert %[[xaddr2vall]] : (i64) -> i32
! CHECK: fir.store %[[xaddr2val]] to %[[p]] : !fir.ref<i32>
end

! CHECK-LABEL: func.func @_QPloc_derived_type() {
subroutine loc_derived_type
  integer(8) :: p
  type dt
    integer :: i
  end type
  type(dt) :: xdt
  p = loc(xdt)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[xdt:.*]] = fir.alloca !fir.type<_QFloc_derived_typeTdt{i:i32}> {{.*}}
! CHECK: %[[xdtbox:.*]] = fir.embox %[[xdt]] : (!fir.ref<!fir.type<_QFloc_derived_typeTdt{i:i32}>>) -> !fir.box<!fir.type<_QFloc_derived_typeTdt{i:i32}>>
! CHECK: %[[xdtaddr:.*]] = fir.box_addr %[[xdtbox]] : (!fir.box<!fir.type<_QFloc_derived_typeTdt{i:i32}>>) -> !fir.ref<!fir.type<_QFloc_derived_typeTdt{i:i32}>>
! CHECK: %[[xdtaddrval:.*]] = fir.convert %[[xdtaddr]] : (!fir.ref<!fir.type<_QFloc_derived_typeTdt{i:i32}>>) -> i64
! CHECK: fir.store %[[xdtaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_pointer_array() {
subroutine loc_pointer_array
  integer(8) :: p
  integer, pointer :: x(:)
  p = loc(x)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[x:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {{.*}}
! CHECK: %2 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK: %[[zero:.*]] = arith.constant 0 : index
! CHECK: %[[xshp:.*]] = fir.shape %[[zero]] : (index) -> !fir.shape<1>
! CHECK: %[[xbox0:.*]] = fir.embox %2(%[[xshp]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK: fir.store %[[xbox0]] to %[[x]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[xbox:.*]] = fir.load %[[x]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK: %[[xaddrval:.*]] = fir.convert %[[xaddr]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK: fir.store %[[xaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_allocatable_array() {
subroutine loc_allocatable_array
  integer(8) :: p
  integer, allocatable :: x(:)
  p = loc(x)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {{.*}}
! CHECK: %[[stg:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {{.*}}
! CHECK: %[[lb:.*]] = fir.alloca index {{.*}}
! CHECK: %[[ext:.*]] = fir.alloca index {{.*}}
! CHECK: %[[zstg:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK: fir.store %[[zstg]] to %[[stg]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK: %[[lbval:.*]] = fir.load %[[lb]] : !fir.ref<index>
! CHECK: %[[extval:.*]] = fir.load %[[ext]] : !fir.ref<index>
! CHECK: %[[stgaddr:.*]] = fir.load %[[stg]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK: %[[ss:.*]] = fir.shape_shift %[[lbval]], %[[extval]] : (index, index) -> !fir.shapeshift<1>
! CHECK: %[[xbox:.*]] = fir.embox %[[stgaddr]](%[[ss]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK: %[[xaddr:.*]] = fir.box_addr %[[xbox]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK: %[[xaddrval:.*]] = fir.convert %[[xaddr]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK: fir.store %[[xaddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPtest_external() {
subroutine test_external()
  integer(8) :: p
  integer, external :: f
  p = loc(x=f)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[f:.*]] = fir.address_of(@_QPf) : () -> i32
! CHECK: %[[fbox:.*]] = fir.emboxproc %[[f]] : (() -> i32) -> !fir.boxproc<() -> i32>
! CHECK: %[[faddr:.*]] = fir.box_addr %[[fbox]] : (!fir.boxproc<() -> i32>) -> (() -> i32)
! CHECK: %[[faddrval:.*]] = fir.convert %[[faddr]] : (() -> i32) -> i64
! CHECK: fir.store %[[faddrval]] to %[[p]] : !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPtest_proc() {
subroutine test_proc()
  integer(8) :: p
  procedure() :: g
  p = loc(x=g)
! CHECK: %[[p:.*]] = fir.alloca i64 {{.*}}
! CHECK: %[[g:.*]] = fir.address_of(@_QPg) : () -> ()
! CHECK: %[[gbox:.*]] = fir.emboxproc %[[g]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK: %[[gaddr:.*]] = fir.box_addr %[[gbox]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK: %[[gaddrval:.*]] = fir.convert %[[gaddr]] : (() -> ()) -> i64
! CHECK: fir.store %[[gaddrval]] to %[[p]] : !fir.ref<i64>
end

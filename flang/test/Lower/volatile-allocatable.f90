! RUN: bbc %s -o - --strict-fir-volatile-verifier | FileCheck %s

! Check that class types, derived types, and polymorphic types can be volatile.
! Check their interaction with allocatable, pointer, and assumed-shape allocatable
! results in values with correctly designated or declared types.

module derived_types
  type :: base_type
    integer :: i = 42
  end type

  type, extends(base_type) :: ext_type
    integer :: j = 100
  end type

  type :: comp_type
    character(10) :: str = "test"
    integer :: arr(2) = [1, 2]
  end type
end module

subroutine test_scalar_volatile()
  use derived_types
  class(base_type), allocatable, volatile :: v1
  type(ext_type), allocatable, volatile :: v2
  type(comp_type), allocatable, volatile :: v3
  character(len=:), allocatable, volatile :: c1

  ! Allocation without source
  allocate(v1)

  ! Allocate polymorphic derived type with dynamic type
  allocate(ext_type :: v1)
  select type (v1)
    type is (ext_type)
      v1%j = 2
  end select

  ! Allocation with source
  allocate(v2, source=ext_type())

  ! Deferred-length characters
  allocate(character(20) :: c1)
  c1 = "volatile character"
  
  ! Allocation with components
  allocate(v3)
  deallocate(v1, v2, v3, c1)
end subroutine

! Test with both volatile and asynchronous attributes
subroutine test_volatile_asynchronous()
  use derived_types
  class(base_type), allocatable, volatile, asynchronous :: v1(:)
  integer, allocatable, volatile, asynchronous :: i1(:)
  
  allocate(v1(4))
  allocate(i1(4), source=[1, 2, 3, 4])
  
  deallocate(v1, i1)
end subroutine

subroutine test_select_base_type_volatile()
  use derived_types
  class(base_type), allocatable, volatile :: v(:)
  
  allocate(v(2))
  
  select type(v)
  class is (base_type)
    v(1)%i = 100
  end select
  
  deallocate(v)
end subroutine

! Test allocate with mold
subroutine test_mold_allocation()
  use derived_types
  type(comp_type) :: template
  type(comp_type), allocatable, volatile :: v(:)
  
  template%str = "mold test"
  template%arr = [5, 6]
  
  allocate(v(3), mold=template)
  
  deallocate(v)
end subroutine

! Test unlimited polymorphic allocation
subroutine test_unlimited_polymorphic()
  use derived_types
  class(*), allocatable, volatile :: up
  class(*), allocatable, volatile :: upa(:)
  
  ! Scalar allocation
  allocate(integer :: up)
  select type(up)
    type is (integer)
      up = 123
  end select
  
  ! Array allocation with source
  allocate(character(10) :: up)
  select type(up)
    type is (character(*))
      up = "class(*)"
  end select
  
  ! Array allocation
  allocate(real :: upa(3))
  select type(upa)
    type is (real)
      upa = [1.1, 2.2, 3.3]
  end select
  
  deallocate(up, upa)
end subroutine

! CHECK-LABEL:   func.func @_QPtest_scalar_volatile() {
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFtest_scalar_volatileEc1"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>, volatile>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFtest_scalar_volatileEv1"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>, !fir.ref<!fir.class<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFtest_scalar_volatileEv2"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFtest_scalar_volatileEv3"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.type<{{.*}}>>, volatile>, volatile>)
! CHECK:           fir.call @_FortranAAllocatableInitDerivedForAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           fir.call @_FortranAAllocatableInitDerivedForAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFtest_scalar_volatileEv1"} : (!fir.box<!fir.heap<!fir.type<{{.*}}>>, volatile>) -> (!fir.box<!fir.type<{{.*}}>, volatile>, !fir.box<!fir.type<{{.*}}>, volatile>)
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0{"j"}   : (!fir.box<!fir.type<{{.*}}>, volatile>) -> !fir.ref<i32, volatile>
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QMderived_typesText_type.0"} : (!fir.ref<!fir.type<{{.*}}>>) -> (!fir.ref<!fir.type<{{.*}}>>, !fir.ref<!fir.type<{{.*}}>>)
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocateSource(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX766F6C6174696C6520636861726163746572"} : (!fir.ref<!fir.char<1,18>>, index) -> (!fir.ref<!fir.char<1,18>>, !fir.ref<!fir.char<1,18>>)
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-LABEL:   func.func @_QPtest_volatile_asynchronous() {
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, asynchronous, volatile>, uniq_name = "_QFtest_volatile_asynchronousEi1"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, asynchronous, volatile>, uniq_name = "_QFtest_volatile_asynchronousEv1"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>)
! CHECK:           fir.call @_FortranAAllocatableInitDerivedForAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.4xi4.1"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocateSource(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-LABEL:   func.func @_QPtest_select_base_type_volatile() {
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFtest_select_base_type_volatileEv"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>)
! CHECK:           fir.call @_FortranAAllocatableInitDerivedForAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAClassIs(%{{.+}}, %{{.+}}) : (!fir.box<none>, !fir.ref<none>) -> i1
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFtest_select_base_type_volatileEv"} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, !fir.shift<1>) -> (!fir.class<!fir.array<?x!fir.type<{{.*}}>>, volatile>, !fir.class<!fir.array<?x!fir.type<{{.*}}>>, volatile>)
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0 (%{{.+}})  : (!fir.class<!fir.array<?x!fir.type<{{.*}}>>, volatile>, index) -> !fir.class<!fir.type<{{.*}}>, volatile>
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}{"i"}   : (!fir.class<!fir.type<{{.*}}>, volatile>) -> !fir.ref<i32, volatile>
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-LABEL:   func.func @_QPtest_mold_allocation() {
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {uniq_name = "_QFtest_mold_allocationEtemplate"} : (!fir.ref<!fir.type<{{.*}}>>) -> (!fir.ref<!fir.type<{{.*}}>>, !fir.ref<!fir.type<{{.*}}>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFtest_mold_allocationEv"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX6D6F6C642074657374"} : (!fir.ref<!fir.char<1,9>>, index) -> (!fir.ref<!fir.char<1,9>>, !fir.ref<!fir.char<1,9>>)
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0{"str"}   typeparams %{{.+}} : (!fir.ref<!fir.type<{{.*}}>>, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.2xi4.2"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
! CHECK:           %{{.+}} = hlfir.designate %{{.+}}#0{"arr"}   shape %{{.+}} : (!fir.ref<!fir.type<{{.*}}>>, !fir.shape<1>) -> !fir.ref<!fir.array<2xi32>>
! CHECK:           fir.call @_FortranAAllocatableApplyMold(%{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK-LABEL:   func.func @_QPtest_unlimited_polymorphic() {
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFtest_unlimited_polymorphicEup"} : (!fir.ref<!fir.class<!fir.heap<none>, volatile>, volatile>) -> (!fir.ref<!fir.class<!fir.heap<none>, volatile>, volatile>, !fir.ref<!fir.class<!fir.heap<none>, volatile>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<allocatable, volatile>, uniq_name = "_QFtest_unlimited_polymorphicEupa"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>, volatile>, volatile>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>, volatile>, volatile>, !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>, volatile>, volatile>)
! CHECK:           fir.call @_FortranAAllocatableInitIntrinsicForAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i32, i32, i32) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFtest_unlimited_polymorphicEup"} : (!fir.heap<i32>) -> (!fir.heap<i32>, !fir.heap<i32>)
! CHECK:           fir.call @_FortranAAllocatableInitCharacterForAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFtest_unlimited_polymorphicEup"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}} typeparams %{{.+}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX636C617373282A29"} : (!fir.ref<!fir.char<1,8>>, index) -> (!fir.ref<!fir.char<1,8>>, !fir.ref<!fir.char<1,8>>)
! CHECK:           fir.call @_FortranAAllocatableInitIntrinsicForAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i32, i32, i32) -> ()
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableAllocate(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFtest_unlimited_polymorphicEupa"} : (!fir.box<!fir.heap<!fir.array<?xf32>>, volatile>, !fir.shift<1>) -> (!fir.box<!fir.array<?xf32>, volatile>, !fir.box<!fir.array<?xf32>, volatile>)
! CHECK:           %{{.+}}:2 = hlfir.declare %{{.+}}(%{{.+}}) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3xr4.3"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %{{.+}} = fir.call @_FortranAAllocatableDeallocatePolymorphic(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

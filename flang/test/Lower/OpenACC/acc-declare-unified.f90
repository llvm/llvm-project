! Test !$acc declare create on allocatables under '-gpu unified'. The variable
! is reachable from device code through its host address, so the allocation and
! deallocation actions of section 2.13.2 must not be generated: mirroring the
! data on the device would create a second copy that the kernels never read.
! The global constructor and destructor are still generated because they map the
! host global onto the symbol that device code references.
!
! The runs without '-gpu unified' are the control: they show the same source
! does generate the actions in discrete memory.

! RUN: split-file %s %t

! RUN: bbc -fopenacc -gpu unified -emit-hlfir %t/mod.f90 -o %t/mod-unified.mlir --module=%t
! RUN: bbc -fopenacc -gpu unified -emit-hlfir %t/use.f90 -o %t/use-unified.mlir -I %t
! RUN: FileCheck %s --check-prefix=UNIFIED-MOD --implicit-check-not=acc.declare_action \
! RUN:   --implicit-check-not=_acc_declare_post_alloc \
! RUN:   --implicit-check-not=_acc_declare_pre_dealloc \
! RUN:   --implicit-check-not=_acc_declare_post_dealloc < %t/mod-unified.mlir
! RUN: FileCheck %s --check-prefix=UNIFIED-USE --implicit-check-not=acc.declare_action \
! RUN:   --implicit-check-not=_acc_declare_post_alloc \
! RUN:   --implicit-check-not=_acc_declare_pre_dealloc \
! RUN:   --implicit-check-not=_acc_declare_post_dealloc < %t/use-unified.mlir

! RUN: bbc -fopenacc -emit-hlfir %t/mod.f90 -o %t/mod-discrete.mlir --module=%t
! RUN: bbc -fopenacc -emit-hlfir %t/use.f90 -o %t/use-discrete.mlir -I %t
! RUN: FileCheck %s --check-prefix=DISCRETE-MOD < %t/mod-discrete.mlir
! RUN: FileCheck %s --check-prefix=DISCRETE-USE < %t/use-discrete.mlir

//--- mod.f90
module acc_declare_unified_mod
  real, allocatable :: garr(:)
  !$acc declare create(garr)
contains
  subroutine alloc_global()
    allocate(garr(10))
    deallocate(garr)
  end subroutine
  subroutine alloc_local()
    real, allocatable :: larr(:)
    !$acc declare create(larr)
    allocate(larr(10))
    deallocate(larr)
  end subroutine
end module

//--- use.f90
subroutine use_mod()
  use acc_declare_unified_mod
  implicit none
  allocate(garr(100))
end subroutine

! The constructor and destructor that map the host global are still emitted.
! UNIFIED-MOD: acc.global_ctor @_QMacc_declare_unified_modEgarr_acc_ctor
! UNIFIED-MOD: acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! UNIFIED-MOD: acc.declare_enter
! UNIFIED-MOD: acc.global_dtor @_QMacc_declare_unified_modEgarr_acc_dtor
! UNIFIED-MOD: acc.declare_exit

! The using unit neither annotates the allocation nor declares external recipes.
! UNIFIED-USE: func.func @_QPuse_mod()
! UNIFIED-USE: fir.global @_QMacc_declare_unified_modEgarr {acc.declare = #acc.declare<dataClause = acc_create>

! DISCRETE-MOD: acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declare_unified_modEgarr_acc_declare_post_alloc>
! DISCRETE-MOD: acc.declare_action = #acc.declare_action<preDealloc = @_QMacc_declare_unified_modEgarr_acc_declare_pre_dealloc>
! DISCRETE-MOD: acc.declare_action = #acc.declare_action<postDealloc = @_QMacc_declare_unified_modEgarr_acc_declare_post_dealloc>
! DISCRETE-MOD: func.func private @_QMacc_declare_unified_modFalloc_localElarr_acc_declare_post_alloc(
! DISCRETE-MOD: func.func private @_QMacc_declare_unified_modFalloc_localElarr_acc_declare_pre_dealloc(
! DISCRETE-MOD: func.func private @_QMacc_declare_unified_modFalloc_localElarr_acc_declare_post_dealloc(
! DISCRETE-MOD: acc.global_ctor @_QMacc_declare_unified_modEgarr_acc_ctor
! DISCRETE-MOD: func.func @_QMacc_declare_unified_modEgarr_acc_declare_post_alloc() attributes {acc.declare_action}
! DISCRETE-MOD: func.func @_QMacc_declare_unified_modEgarr_acc_declare_post_dealloc() attributes {acc.declare_action}
! DISCRETE-MOD: acc.global_dtor @_QMacc_declare_unified_modEgarr_acc_dtor

! DISCRETE-USE: acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declare_unified_modEgarr_acc_declare_post_alloc>
! DISCRETE-USE: func.func private @_QMacc_declare_unified_modEgarr_acc_declare_post_alloc()
! DISCRETE-USE: func.func private @_QMacc_declare_unified_modEgarr_acc_declare_post_dealloc()

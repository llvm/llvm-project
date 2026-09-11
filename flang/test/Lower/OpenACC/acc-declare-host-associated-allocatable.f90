! Test !$acc declare create on an allocatable that is allocated and deallocated
! from an internal subprogram. The OpenACC flags are set on the host scope
! symbol, so the host associated symbol seen at the ALLOCATE and DEALLOCATE
! statements has to be resolved before the declare actions are attached.
! Covers both the runtime call path (derived type) and the inlined path
! (intrinsic type).

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir %s -o - | fir-opt --acc-declare-action-conversion | \
! RUN:   FileCheck %s --check-prefix=CONV

program acc_declare_host_assoc
  type domain_type
     integer :: nvar
  end type domain_type

  type(domain_type), allocatable :: domains(:)
  integer, allocatable :: data(:)
  !$acc declare create(domains, data)

  call init
  call fini

contains

  subroutine init()
    allocate(domains(10))
    allocate(data(10))
  end subroutine init

  subroutine fini()
    deallocate(domains)
    deallocate(data)
  end subroutine fini

end program acc_declare_host_assoc

! CHECK-LABEL: func.func private @_QFPinit()
! CHECK: fir.call @_FortranAAllocatableAllocate({{.*}}) fastmath<contract> {acc.declare_action = #acc.declare_action<postAlloc = @_QFEdomains_acc_declare_post_alloc>}
! CHECK: fir.allocmem !fir.array<?xi32>
! CHECK: fir.store %{{.*}} to %{{.*}} {acc.declare_action = #acc.declare_action<postAlloc = @_QFEdata_acc_declare_post_alloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

! CHECK-LABEL: func.func private @_QFPfini()
! CHECK: fir.call @_FortranAAllocatableDeallocate({{.*}}) fastmath<contract> {acc.declare_action = #acc.declare_action<preDealloc = @_QFEdomains_acc_declare_pre_dealloc, postDealloc = @_QFEdomains_acc_declare_post_dealloc>}
! CHECK: fir.box_addr %{{.*}} {acc.declare_action = #acc.declare_action<preDealloc = @_QFEdata_acc_declare_pre_dealloc>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: fir.freemem
! CHECK: fir.store %{{.*}} to %{{.*}} {acc.declare_action = #acc.declare_action<postDealloc = @_QFEdata_acc_declare_post_dealloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

! The attributes must be actionable: the conversion pass has to find each
! recipe and insert the call next to the allocation or deallocation.

! CONV-LABEL: func.func private @_QFPinit()
! CONV: fir.call @_FortranAAllocatableAllocate(
! CONV: fir.call @_QFEdomains_acc_declare_post_alloc(%{{.*}}) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFTdomain_type{nvar:i32}>>>>>) -> ()
! CONV: fir.call @_QFEdata_acc_declare_post_alloc(%{{.*}}) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()

! CONV-LABEL: func.func private @_QFPfini()
! CONV: fir.call @_QFEdomains_acc_declare_pre_dealloc(%{{.*}}) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFTdomain_type{nvar:i32}>>>>>) -> ()
! CONV: fir.call @_FortranAAllocatableDeallocate(
! CONV: fir.call @_QFEdomains_acc_declare_post_dealloc(%{{.*}}) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFTdomain_type{nvar:i32}>>>>>) -> ()
! CONV: fir.call @_QFEdata_acc_declare_pre_dealloc(%{{.*}}) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()
! CONV: fir.freemem
! CONV: fir.call @_QFEdata_acc_declare_post_dealloc(%{{.*}}) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()

! RUN: %python %S/test_errors.py %s %flang_fc1
! RUN: %flang_fc1 -fsyntax-only -Wpointer-to-undefinable %s 2>&1 | FileCheck %s --implicit-check-not="Pointer target is not a definable"
! Positive tests for Fortran 2028 PROTECTED_TARGET

module m_prot
  integer, target, protected :: prot_target = 42
end module

module m_pt_positive
  use m_prot
  type sub_t
    integer :: a = 10
    integer, pointer :: pc => null()
  end type

  type container_t
    type(sub_t), pointer, protected_target :: ptc => null()
  end type

  integer, target :: mod_target = 100
  integer, pointer, protected_target :: mod_pt
  integer, pointer, protected_target :: init_pt => prot_target

contains
  function get_pt() result(r)
    integer, pointer, protected_target :: r
    r => mod_target
  end function

  subroutine test_positive()
    integer, target :: local_val = 1
    integer, target :: local_arr(3) = [1, 2, 3]
    type(sub_t), target :: local_sub
    integer, pointer, protected_target :: p
    integer, pointer, protected_target :: p_arr(:)
    integer, pointer, protected_target :: q
    class(*), pointer, protected_target :: poly_pt
    type(sub_t), pointer, protected_target :: p_sub
    type(container_t) :: cont
    integer, pointer :: ord_ptr
    integer :: val
    logical :: l_assoc

    ! 1. Pointer association to target (local and module)
    p => local_val
    mod_pt => mod_target
    val = p + mod_pt

    ! 2. 8.5.16 NOTE 1 vs C864: Pointer association to use-associated protected target
    ! Draft J3/26-007r1 C864 forbids a use-associated nonpointer PROTECTED object
    ! as a data-target, while 8.5.16 NOTE 1 shows it as valid for PROTECTED_TARGET.
    ! Following NOTE 1, ordinary pointer gets warning under -Wpointer-to-undefinable;
    ! PROTECTED_TARGET pointer does not (in pointer-assignment or initial-data-target).
    ! CHECK: protected_target02.f90:[[@LINE+2]]:{{[0-9]+}}: warning: Pointer target is not a definable variable
    ! CHECK: because: 'prot_target' is protected in this scope
    ord_ptr => prot_target
    p => prot_target
    val = p + init_pt

    ! Conforming NULL(MOLD=) with PROTECTED_TARGET pointer
    p => null(p)
    ord_ptr => null(p)

    ! 3. Allocation with SOURCE=
    allocate(p, source=99)
    nullify(p)

    ! 4. Pointer-to-pointer assignment preserving protection
    q => p
    p_arr => local_arr
    q => p_arr(2)

    ! 5. Calling procedure with matching PROTECTED_TARGET pointer dummy
    call take_pt_ptr(p)

    ! 6. Calling procedure with INTENT(IN) non-pointer dummy
    call take_intent_in(p)
    call take_intent_in(p_arr(1))

    ! 7. Calling procedure with pointer INTENT(OUT) dummy
    call take_pt_ptr_out(p)

    ! 8. Structure constructor with PROTECTED_TARGET component
    cont = container_t(p_sub)

    ! 9. Function result with PROTECTED_TARGET
    q => get_pt()

    ! 10. Pointer component of target is ordinary pointer: can define its target.
    ! Under J3/26-007r1 9.4.2p5 and NOTE 1, the target of X%P is not a subobject
    ! of X. The pointer X%P is a subobject of X only in contexts that pertain
    ! to its pointer association status. Thus, the target of p_sub%pc can be
    ! defined or associated with an ordinary pointer.
    p_sub%pc = 200
    ord_ptr => p_sub%pc
    val = ord_ptr

    ! 11. Component pointer association and allocation
    cont%ptc => local_sub
    nullify(cont%ptc)
    allocate(cont%ptc, source=local_sub)

    ! 12. Target of an ordinary pointer component passed to non-pointer dummy.
    ! Under 9.4.2p5 and NOTE 1, the target of p_sub%pc is not a subobject of p_sub,
    ! so passing it in a non-pointer context is conforming.
    call take_no_intent(p_sub%pc)

    ! 13. Polymorphic PROTECTED_TARGET pointer in SELECT TYPE read
    poly_pt => local_val
    select type (a => poly_pt)
    type is (integer)
      val = a
    end select

    ! 14. Target of an ordinary pointer component through ASSOCIATE.
    ! Under 9.4.2p5 and NOTE 1, pc is the nearest pointer in p_sub%pc, so its
    ! target is not protected. Construct association preserves this.
    associate (b => p_sub%pc)
      ord_ptr => b
      b = 3
    end associate
    associate (c => p_sub)
      ord_ptr => c%pc
      c%pc = 4
    end associate

    ! 15. Intrinsic inquiry ASSOCIATED accepts PROTECTED_TARGET pointer
    l_assoc = associated(p)
    l_assoc = associated(ord_ptr, p)
  end subroutine

  subroutine take_pt_ptr(arg)
    integer, pointer, protected_target :: arg
    integer :: x
    x = arg
  end subroutine

  subroutine take_intent_in(arg)
    integer, intent(in) :: arg
    integer :: x
    x = arg
  end subroutine

  subroutine take_pt_ptr_out(arg)
    integer, pointer, protected_target, intent(out) :: arg
    nullify(arg)
  end subroutine

  subroutine take_no_intent(arg)
    integer :: arg
    integer :: x
    x = arg
  end subroutine
end module

module m_pt_rename
  use m_prot, only: ren_prot => prot_target
contains
  subroutine test_rename()
    integer, pointer, protected_target :: p
    integer :: val
    p => ren_prot
    val = p
  end subroutine
end module

! C867 applies to pointers named in a COMMON statement; a sequence-type object
! with a PROTECTED_TARGET pointer component in COMMON is accepted.
subroutine test_common_seq()
  type seq_t
    sequence
    integer, pointer, protected_target :: pc
  end type
  type(seq_t) :: obj
  common /cc/ obj
end subroutine

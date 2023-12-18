! RUN: %python %S/test_errors.py %s %flang_fc1
! Test 15.5.2.5 constraints and restrictions for POINTER & ALLOCATABLE
! arguments when both sides of the call have the same attributes.

module m

  type :: t
  end type
  type, extends(t) :: t2
  end type
  type :: pdt(n)
    integer, len :: n
  end type

  type(t), pointer :: mp(:), mpmat(:,:)
  type(t), allocatable :: ma(:), mamat(:,:)
  class(t), pointer :: pp(:)
  class(t), allocatable :: pa(:)
  class(t2), pointer :: pp2(:)
  class(t2), allocatable :: pa2(:)
  class(*), pointer :: up(:)
  class(*), allocatable :: ua(:)
  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, character named constant, or external function result
  type(pdt(*)), pointer :: amp(:)
  !ERROR: An assumed (*) type parameter may be used only for a (non-statement function) dummy argument, associate name, character named constant, or external function result
  type(pdt(*)), allocatable :: ama(:)
  type(pdt(:)), pointer :: dmp(:)
  type(pdt(:)), allocatable :: dma(:)
  type(pdt(1)), pointer :: nmp(:)
  type(pdt(1)), allocatable :: nma(:)

 contains

  subroutine smp(x)
    type(t), pointer :: x(:)
  end subroutine
  subroutine sma(x)
    type(t), allocatable :: x(:)
  end subroutine
  subroutine spp(x)
    class(t), pointer :: x(:)
  end subroutine
  subroutine spa(x)
    class(t), allocatable :: x(:)
  end subroutine
  subroutine sup(x)
    class(*), pointer :: x(:)
  end subroutine
  subroutine sua(x)
    class(*), allocatable :: x(:)
  end subroutine
  subroutine samp(x)
    type(pdt(*)), pointer :: x(:)
  end subroutine
  subroutine sama(x)
    type(pdt(*)), allocatable :: x(:)
  end subroutine
  subroutine sdmp(x)
    type(pdt(:)), pointer :: x(:)
  end subroutine
  subroutine sdma(x)
    type(pdt(:)), allocatable :: x(:)
  end subroutine
  subroutine snmp(x)
    type(pdt(1)), pointer :: x(:)
  end subroutine
  subroutine snma(x)
    type(pdt(1)), allocatable :: x(:)
  end subroutine

  subroutine test
    call smp(mp) ! ok
    call sma(ma) ! ok
    call spp(pp) ! ok
    call spa(pa) ! ok
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so
    call smp(pp)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so
    call sma(pa)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so
    call spp(mp)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so
    call spa(ma)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both must be so
    call sup(pp)
    !ERROR: If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both must be so
    call sua(pa)
    !ERROR: Actual argument type 'CLASS(*)' is not compatible with dummy argument type 'CLASS(t)'
    !ERROR: Pointer type must be unlimited polymorphic or non-extensible derived type when target is unlimited polymorphic
    call spp(up)
    !ERROR: Actual argument type 'CLASS(*)' is not compatible with dummy argument type 'CLASS(t)'
    call spa(ua)
    !ERROR: POINTER or ALLOCATABLE dummy and actual arguments must have the same declared type and kind
    call spp(pp2)
    !ERROR: POINTER or ALLOCATABLE dummy and actual arguments must have the same declared type and kind
    call spa(pa2)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    !ERROR: Pointer has rank 1 but target has rank 2
    call smp(mpmat)
    !ERROR: Rank of dummy argument is 1, but actual argument has rank 2
    call sma(mamat)
    call sdmp(dmp) ! ok
    call sdma(dma) ! ok
    call snmp(nmp) ! ok
    call snma(nma) ! ok
    call samp(nmp) ! ok
    call sama(nma) ! ok
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call sdmp(nmp)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call sdma(nma)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call snmp(dmp)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call snma(dma)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call samp(dmp)
    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call sama(dma)
  end subroutine

end module

module m2

  !ERROR: Procedure 't3' may not be ALLOCATABLE without an explicit interface
  character(len=10), allocatable :: t1, t2, t3, t4
  !ERROR: Procedure 't6' may not be ALLOCATABLE without an explicit interface
  character(len=:), allocatable :: t5, t6, t7, t8(:)

  character(len=10), pointer :: p1
  character(len=:), pointer :: p2

  integer, allocatable :: x(:)

 contains

  subroutine sma(a)
    character(len=:), allocatable, intent(in) :: a
  end

  subroutine sma2(a)
    character(len=10), allocatable, intent(in) :: a
  end

  subroutine smp(p)
    character(len=:), pointer, intent(in) :: p
  end

  subroutine smp2(p)
    character(len=10), pointer, intent(in) :: p
  end

  subroutine smb(b)
    integer, allocatable, intent(in) :: b(:)
  end

  function return_deferred_length_ptr()
    character(len=:), pointer :: return_deferred_length_ptr
  end function

  function return_explicit_length_ptr(n)
    integer :: n
    character(len=n), pointer :: return_explicit_length_ptr
  end function

  subroutine test()

    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call sma(t1)

    call sma2(t1) ! ok

    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call smp(p1)

    call smp2(p1) ! ok

    call smp(return_deferred_length_ptr()) ! ok

    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call smp2(return_deferred_length_ptr())

    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call smp(return_explicit_length_ptr(10))

    call smp2(return_explicit_length_ptr(10)) ! ok

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call sma(t2(:))

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call sma(t3(1))

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call sma(t4(1:2))

    call sma(t5) ! ok

    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call sma2(t5)

    call smp(p2) ! ok

    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call smp2(p2)

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call sma(t5(:))

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call sma(t6(1))

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call sma(t7(1:2))

    !ERROR: ALLOCATABLE dummy argument 'a=' must be associated with an ALLOCATABLE actual argument
    call sma(t8(1))

    !ERROR: ALLOCATABLE dummy argument 'b=' must be associated with an ALLOCATABLE actual argument
    call smb(x(:))

    !ERROR: Rank of dummy argument is 1, but actual argument has rank 0
    !ERROR: ALLOCATABLE dummy argument 'b=' must be associated with an ALLOCATABLE actual argument
    call smb(x(2))

    !ERROR: ALLOCATABLE dummy argument 'b=' must be associated with an ALLOCATABLE actual argument
    call smb(x(1:2))

  end subroutine

end module

module test
  type t(l)
    integer, len :: l
    character(l) :: c
  end type

 contains

  subroutine bar(p)
    type(t(:)), allocatable :: p(:)
  end subroutine

  subroutine foo
    type(t(10)), allocatable :: p(:)

    !ERROR: Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE
    call bar(p)

  end subroutine

end module

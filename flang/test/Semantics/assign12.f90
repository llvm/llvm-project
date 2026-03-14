! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type base
    procedure(baseSub), pointer :: baseComponent
  end type
  type, extends(base) :: extended
  end type
 contains
  subroutine baseSub(x)
    class(base), intent(in) :: x
  end
  subroutine extendedSub(x)
    class(extended), intent(in) :: x
  end
  subroutine baseSubmono(x)
    type(base), intent(in) :: x
  end
  subroutine test
    procedure(baseSub), pointer :: basePtr
    procedure(extendedSub), pointer :: extendedPtr
    type(extended) :: extendedVar
    extendedPtr => baseSub ! ok
    extendedPtr => basePtr ! ok
    extendedVar = extended(baseSub) ! ok
    extendedVar = extended(basePtr) ! ok
    !ERROR: Procedure pointer 'baseptr' associated with incompatible procedure designator 'extendedsub': incompatible dummy argument #1: incompatible dummy data object types: CLASS(extended) vs CLASS(base)
    basePtr => extendedSub
    !ERROR: Procedure pointer 'baseptr' associated with incompatible procedure designator 'extendedptr': incompatible dummy argument #1: incompatible dummy data object types: CLASS(extended) vs CLASS(base)
    basePtr => extendedPtr
    !ERROR: Procedure pointer 'basecomponent' associated with incompatible procedure designator 'extendedsub': incompatible dummy argument #1: incompatible dummy data object types: CLASS(extended) vs CLASS(base)
    extendedVar = extended(extendedSub)
    !ERROR: Procedure pointer 'basecomponent' associated with incompatible procedure designator 'extendedptr': incompatible dummy argument #1: incompatible dummy data object types: CLASS(extended) vs CLASS(base)
    extendedVar = extended(extendedPtr)
    !ERROR: Procedure pointer 'baseptr' associated with incompatible procedure designator 'basesubmono': incompatible dummy argument #1: incompatible dummy data object polymorphism: base vs CLASS(base)
    basePtr => baseSubmono
  end
end

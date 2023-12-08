! RUN: %python %S/test_errors.py %s %flang_fc1
module m

  ! For C1543
  interface intFace
    !WARNING: Attribute 'MODULE' cannot be used more than once
    module pure module real function moduleFunc()
    end function moduleFunc
  end interface

contains

! C1543 A prefix shall contain at most one of each prefix-spec.
! 
! R1535 subroutine-stmt is 
!   [prefix] SUBROUTINE subroutine-name [ ( [dummy-arg-list] ) 
!   [proc-language-binding-spec] ]
! 
! R1526  prefix is
!   prefix-spec[prefix-spec]...
!   
!   prefix-spec values are:
!      declaration-type-spec, ELEMENTAL, IMPURE, MODULE, NON_RECURSIVE, 
!      PURE, RECURSIVE

    !ERROR: FUNCTION prefix cannot specify the type more than once
    real pure real function realFunc()
    end function realFunc

    !WARNING: Attribute 'ELEMENTAL' cannot be used more than once
    elemental real elemental function elementalFunc(x)
      real, value :: x
      elementalFunc = x
    end function elementalFunc

    !WARNING: Attribute 'IMPURE' cannot be used more than once
    impure real impure function impureFunc()
    end function impureFunc

    !WARNING: Attribute 'PURE' cannot be used more than once
    pure real pure function pureFunc()
    end function pureFunc

    !ERROR: Attributes 'PURE' and 'IMPURE' conflict with each other
    impure real pure function impurePureFunc()
    end function impurePureFunc

    !WARNING: Attribute 'RECURSIVE' cannot be used more than once
    recursive real recursive function recursiveFunc()
    end function recursiveFunc

    !WARNING: Attribute 'NON_RECURSIVE' cannot be used more than once
    non_recursive real non_recursive function non_recursiveFunc()
    end function non_recursiveFunc

    !ERROR: Attributes 'RECURSIVE' and 'NON_RECURSIVE' conflict with each other
    non_recursive real recursive function non_recursiveRecursiveFunc()
    end function non_recursiveRecursiveFunc
end module m

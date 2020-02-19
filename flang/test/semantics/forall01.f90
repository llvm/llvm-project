subroutine forall1
  real :: a(9)
  !ERROR: 'i' is already declared in this scoping unit
  !ERROR: Cannot redefine FORALL variable 'i'
  forall (i=1:8, i=1:9)  a(i) = i
  !ERROR: 'i' is already declared in this scoping unit
  !ERROR: Cannot redefine FORALL variable 'i'
  forall (i=1:8, i=1:9)
    a(i) = i
  end forall
  forall (j=1:8)
    !ERROR: 'j' is already declared in this scoping unit
    !ERROR: Cannot redefine FORALL variable 'j'
    forall (j=1:9)
    end forall
  end forall
end


subroutine forall2
  integer, pointer :: a(:)
  integer, target :: b(10,10)
  forall (i=1:10)
    !ERROR: Impure procedure 'f_impure' may not be referenced in a FORALL
    a(f_impure(i):) => b(i,:)
  end forall
  !ERROR: FORALL mask expression may not reference impure procedure 'f_impure'
  forall (j=1:10, f_impure(1)>2)
  end forall
contains
  impure integer function f_impure(i)
    f_impure = i
  end
end

subroutine forall3
  real :: x
  forall(i=1:10)
    !ERROR: Cannot redefine FORALL variable 'i'
    i = 1
  end forall
  forall(i=1:10)
    forall(j=1:10)
      !ERROR: Cannot redefine FORALL variable 'i'
      i = 1
    end forall
  end forall
  !ERROR: Cannot redefine FORALL variable 'i'
  forall(i=1:10) i = 1
end

subroutine forall4
  integer, parameter :: zero = 0
  integer :: a(10)

  !ERROR: FORALL limit expression may not reference index variable 'i'
  forall(i=1:i)
    a(i) = i
  end forall
  !ERROR: FORALL step expression may not reference index variable 'i'
  forall(i=1:10:i)
    a(i) = i
  end forall
  !ERROR: FORALL step expression may not be zero
  forall(i=1:10:zero)
    a(i) = i
  end forall

  !ERROR: FORALL limit expression may not reference index variable 'i'
  forall(i=1:i) a(i) = i
  !ERROR: FORALL step expression may not reference index variable 'i'
  forall(i=1:10:i) a(i) = i
  !ERROR: FORALL step expression may not be zero
  forall(i=1:10:zero) a(i) = i
end

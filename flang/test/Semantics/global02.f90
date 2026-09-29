! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! Catch discrepancies between implicit result types and a global definition

complex function zbefore()
zbefore = (0.,0.)
end

program main
!ERROR: Implicit declaration of function 'zbefore' has a different result type than in previous declaration
print *, zbefore()
print *, zafter()
print *, zafter2()
print *, zafter3()
end

subroutine another
implicit integer(z)
!ERROR: Implicit declaration of function 'zafter' has a different result type than in previous declaration
print *, zafter()
end

!ERROR: Function 'zafter' has a result type that differs from the implicit type it obtained in a previous reference
complex function zafter()
zafter = (0.,0.)
end

function zafter2()
!ERROR: Function 'zafter2' has a result type that differs from the implicit type it obtained in a previous reference
complex zafter2
zafter2 = (0.,0.)
end

function zafter3() result(res)
!ERROR: Function 'zafter3' has a result type that differs from the implicit type it obtained in a previous reference
complex res
res = (0.,0.)
end

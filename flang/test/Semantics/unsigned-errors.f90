! RUN: %python %S/test_errors.py %s %flang_fc1 -funsigned

implicit unsigned(u)
real a(10)

!ERROR: Must have INTEGER type, but is UNSIGNED(4)
real(kind=4u) x

!ERROR: Both operands must be UNSIGNED
print *, 0 + 1u
!ERROR: Both operands must be UNSIGNED
print *, 0u + 1
!ERROR: Both operands must be UNSIGNED
print *, 0. + 1u
!ERROR: Both operands must be UNSIGNED
print *, 0u + 1.

print *, -0u ! ok
print *, 0u + 1u ! ok
print *, 0u - 1u ! ok
print *, 0u * 1u ! ok
print *, 0u / 1u ! ok
print *, 0u ** 1u ! ok

print *, uint((0.,0.)) ! ok
print *, uint(z'123') ! ok
!ERROR: Actual argument for 'a=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
print *, uint("a")
!ERROR: Actual argument for 'a=' has bad type 'LOGICAL(4)'
print *, uint(.true.)
!ERROR: Actual argument for 'l=' has bad type 'UNSIGNED(4)'
print *, logical(0u)
!ERROR: Actual argument for 'i=' has bad type 'UNSIGNED(4)'
print *, char(0u)

!ERROR: DO controls should be INTEGER
!ERROR: DO controls should be INTEGER
!ERROR: DO controls should be INTEGER
do u = 0u, 1u
end do
!ERROR: DO controls should be INTEGER
do u = 0, 1
end do
!ERROR: DO controls should be INTEGER
!ERROR: DO controls should be INTEGER
do j = 0u, 1u
end do

select case (u) ! ok
case(0u) ! ok
!ERROR: CASE value has type 'INTEGER(4)' which is not compatible with the SELECT CASE expression's type 'UNSIGNED(4)'
case(1)
end select

select case (j)
!ERROR: CASE value has type 'UNSIGNED(4)' which is not compatible with the SELECT CASE expression's type 'INTEGER(4)'
case(0u)
end select

u = z'1' ! ok
!ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types UNSIGNED(4) and INTEGER(4)
u = 1
!ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types INTEGER(4) and UNSIGNED(4)
j = 1u

!ERROR: I/O unit must be a character variable or a scalar integer expression, but is an expression of type UNSIGNED(4)
write(6u,*) 'hi'

!ERROR: ARITHMETIC IF expression must not be an UNSIGNED expression
if (1u) 1,1,1
1 continue

!ERROR: Must have INTEGER type, but is UNSIGNED(4)
print *, a(u)

end

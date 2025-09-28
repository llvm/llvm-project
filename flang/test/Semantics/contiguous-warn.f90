! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
integer, parameter :: num = 3
integer, parameter :: arr(num)=[(i, i=1,num)]
!WARNING: constant values constructed at compile time are likely to be contiguous [-Wconstant-is-contiguous]
logical, parameter :: result=is_contiguous(arr(num:1:-1))
end

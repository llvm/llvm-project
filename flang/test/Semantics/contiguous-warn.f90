! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
integer, parameter :: num = 3
integer, parameter :: arr(num)=[(i, i=1,num)]
!WARNING: is_contiguous() is always true for named constants and subobjects of named constants [-Wconstant-is-contiguous]
logical, parameter :: result=is_contiguous(arr(num:1:-1))
end

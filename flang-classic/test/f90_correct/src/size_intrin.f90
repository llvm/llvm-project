! Call to "size" intrinsic is inlined
program size_intrin
  integer :: array(10) = 0
  integer , parameter :: n = 3
  integer :: result(n) , expect(n)
  expect(1) = 10
  expect(2) = 10
  expect(3) = 0
  result(1) = size(array(1:10:1),1)
  result(2) = size(array(10:1:-1),1)
  result(3) = size(array(1:10:-1),1)
  call checkf(result,expect,n)
end program size_intrin

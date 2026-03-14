! RUN: %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK-NOT: do [1-9]

! Figure out how to also execute this test.

program main
  integer :: results(100)
  integer :: count
  count = 0
  if (.true.) then
    do 1 j1=1,2
      count = count + 1
      results(count) = j1
1   continue
  end if
  do 2 j1=3,4
    do 2 j2=1,2
      if (j1 == j2) then
        do 3 j3=1,2
          count = count + 1
          results(count) = 100*j1 + 10*j2 + j3
          do 3 j4=1,2
            do
              count = count + 1
              results(count) = 10*j3 + j4
              exit
            end do
3         end do
      else
        do
          do 4 j3=3,4
            count = count + 1
            results(count) = 100*j1 + 10*j2 + j3
            do 4 j4=3,4
              count = count + 1
              results(count) = 10*j3 + j4
4           end do
          exit
        end do
      end if
    count = count + 1
    results(count) = 10*j1 + j2
2   continue
  do 5 j1=5,6 ! adjacent non-block DO loops
    count = count + 1
5   results(count) = j1
  do 6 j1=7,8 ! non-block DO loop at end of execution part
    count = count + 1
6   results(count) = j1
  if (count == 34 .and. sum(results(1:count)) == 3739) then
    print *, 'pass'
  else
    print *, 'FAIL:', count, sum(results(1:count)), results(1:count)
  end if
end

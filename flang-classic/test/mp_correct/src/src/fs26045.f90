! test case contributed by Janus Weil with flang issue #528
program omp_associate

integer, parameter :: N = 10000

type tBox
   integer :: cnt = 0
end type

integer ok
type(tBox), dimension(1:N) :: boxes

ok = 0
!$omp parallel do default(shared) private(i)
do i = 1,N
   associate(box => boxes(i))
      if (box%cnt>0) then
!         print *, i, box%cnt
         ok = ok + 1
      else
         box%cnt = i
      end if
   end associate
end do
!$omp end parallel do

if (ok.gt.0) then
  print *, "FAIL"
else
  print *, "PASS"
endif

end

! test contributed by Pawel Osmialowski from ARM
! related to FLANG github issue #474
program workshare
  implicit none
  integer :: somevals(65535)

  !$OMP PARALLEL WORKSHARE
    somevals(:) = 0.0
  !$OMP END PARALLEL WORKSHARE
  print *,"PASS"
end program

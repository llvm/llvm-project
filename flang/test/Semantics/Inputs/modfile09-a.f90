module m
  integer :: m1_x
  interface
    module subroutine s()
    end subroutine
  end interface
end

!Expect: m.mod
!module m
!integer(4)::m1_x
!interface
!module subroutine s()
!end
!end interface
!end

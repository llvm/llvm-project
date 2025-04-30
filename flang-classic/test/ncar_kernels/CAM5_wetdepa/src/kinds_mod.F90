module kinds_mod

 integer, public, parameter :: i4 = selected_int_kind ( 6)  ! 4 byte integer
 integer, public, parameter :: r4 = selected_real_kind ( 6) ! 4 byte real
 integer, public, parameter :: r8 = selected_real_kind (12) ! 8 byte real


end module kinds_mod

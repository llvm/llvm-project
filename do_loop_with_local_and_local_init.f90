! For testing try: `flang -fc1 -emit-hlfir -mmlir --openmp-enable-delayed-privatization-staging=true do_loop_with_local_and_local_init.f90 -o test.mlir

! TODO Will be added as proper test later.
subroutine omploop
  implicit none
  integer :: i, local_var, local_init_var

  do concurrent (i=1:10) local(local_var) local_init(local_init_var)
    if (i < 5) then
      local_var = 42
    else 
      local_init_var = 84
    end if
  end do
end subroutine

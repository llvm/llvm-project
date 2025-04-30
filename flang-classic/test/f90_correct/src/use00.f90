module f_precisions
  integer, parameter, public :: f_double = selected_real_kind(15, 307)
end module f_precisions

module box
  use f_precisions, gp=>f_double
  private
  public :: get_gp
contains
  subroutine get_gp(out_val)
    integer, intent(out) :: out_val
    out_val = gp
  end subroutine get_gp
end module box

module Poisson_Solver
  use f_precisions
  integer, parameter, public :: gp=f_double
end module Poisson_Solver

module module_defs
  use f_precisions
  integer, parameter, public :: gp=f_double
end module module_defs

subroutine PSolver(eh)
  use module_defs ! This has a public gp
  use box ! This has a private gp
  use Poisson_Solver, except_gp => gp ! This has a public gp
  real(gp), intent(out) :: eh ! Should be able to use gp from module_base
end subroutine PSolver

program main
  use box
  use Poisson_Solver
  integer :: box_gp
  call get_gp(box_gp)
  call check(box_gp, gp, 1) ! box's gp and Solver's gp alias the same variable
end program

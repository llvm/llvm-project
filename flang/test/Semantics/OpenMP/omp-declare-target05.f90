! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.14.7 Declare Target Directive

module mod0
  integer :: mi

contains
  subroutine subm()
    integer, save :: mmi

    !ERROR: The DECLARE TARGET directive and the common block or variable in it must appear in the same declaration section of a scoping unit
    !$omp declare target (mi)
    mi = 1
  contains
    subroutine subsubm()
      !ERROR: The DECLARE TARGET directive and the common block or variable in it must appear in the same declaration section of a scoping unit
      !$omp declare target (mmi)
    end
  end
end

module mod1
  integer :: mod_i
end

program main
  use mod1
  integer, save :: i
  integer :: j

  !ERROR: The DECLARE TARGET directive and the common block or variable in it must appear in the same declaration section of a scoping unit
  !$omp declare target (mod_i)

contains
  subroutine sub()
    !ERROR: The DECLARE TARGET directive and the common block or variable in it must appear in the same declaration section of a scoping unit
    !ERROR: The DECLARE TARGET directive and the common block or variable in it must appear in the same declaration section of a scoping unit
    !$omp declare target (i, j)
    i = 1
    j = 1
  end
end

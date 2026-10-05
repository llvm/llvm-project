module m
contains
  subroutine twice(x, y)
    double precision, intent(in) :: x
    double precision, intent(out) :: y
    y = 2 * x
  end subroutine twice
end module m

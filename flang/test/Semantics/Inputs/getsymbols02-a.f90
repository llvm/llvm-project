module mm2a
implicit none
private
  public :: get5
contains
  function get5() result(ret)
    integer :: ret
    ret = 5
    end function get5
end module mm2a

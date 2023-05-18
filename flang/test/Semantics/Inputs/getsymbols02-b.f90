module mm2b
use mm2a
implicit none
private
  public :: callget5
contains
  function callget5() result(ret)
    implicit none
    INTEGER :: ret
    ret = get5()
  end function callget5
end module mm2b

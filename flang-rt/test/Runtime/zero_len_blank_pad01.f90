! Test intrinsic assignments with zero-len source arrays.
! The resulting LHS should be blank-padded - not retain its original value.
! UNSUPPORTED: offload-cuda
! RUN: %flang -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
! CHECK: #                                        #
! CHECK: PASS
program zero_len_blank_pad01
  implicit none
  character(len=0), dimension(0)  :: zla
  character(len=4), dimension(10) :: dest
  zla = ""
  dest = "ABCD"
  dest(1:) = zla(called(1):)(:called(0))
  print *, "#", dest, "#"
  print *, "PASS"

contains

  pure integer function called(i)
    integer, intent(in) :: i
    called = i
  END

end


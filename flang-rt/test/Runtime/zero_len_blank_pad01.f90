! Test intrinsic assignments with zero-len source arrays.
! The resulting LHS should be blank-padded - not retain its original value.
! UNSUPPORTED: offload-cuda
! RUN: %flang -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
! CHECK: #                                        #
! CHECK: PASS
program zero_len_blank_pad01
  INTEGER CALLED
  CHARACTER (LEN=0) , DIMENSION(0)  :: ZEROZERO
  CHARACTER (LEN=4),  DIMENSION(10) :: TENTEN
  ZEROZERO = ""
  TENTEN = "ABCD"
  TENTEN(1:) = ZEROZERO(CALLED(1):)(:CALLED(0))
  print *, "#", TENTEN, "#"
  print *, "PASS"
end

INTEGER FUNCTION CALLED(I)
  integer, intent(in) :: I
  CALLED = I
END

! RUN: %S/test_errors.sh %s %t %f18
program main
  implicit none
  integer :: i = -1
  integer, pointer :: p_i
  integer(kind = 1) :: invalid = 0
  integer, dimension(1:100) :: iarray
  integer, dimension(:), pointer :: p_iarray
  integer, allocatable, dimension(:) :: aiarray
  logical :: l = .false.
  logical, dimension(1:100) :: larray
  logical, allocatable, dimension(:) :: alarray
  character(len = 128) :: chr1
  character(kind = 4, len = 128) :: chr2

  if (i .eq. 0) stop "Stop."
  if (i .eq. 0) stop "Stop."(1:4)
  if (i .eq. 0) stop chr1
!ERROR: CHARACTER stop code must be of default kind
  if (i .eq. 0) stop chr2
  if (i .eq. 0) stop 1
  if (i .eq. 0) stop 1 + 2
  if (i .eq. 0) stop i
  if (i .eq. 0) stop p_i
  if (i .eq. 0) stop p_iarray(1)
  if (i .eq. 0) stop iarray(1)
  if (i .eq. 0) stop aiarray(1)
  if (i .eq. 0) stop 1 + i
!ERROR: INTEGER stop code must be of default kind
  if (i .eq. 0) stop invalid
!ERROR: Stop code must be of INTEGER or CHARACTER type
  if (i .eq. 0) stop 12.34
  if (i .eq. 0) stop 1, quiet = .true.
  if (i .eq. 0) stop 2, quiet = .false.
  if (i .eq. 0) stop 3, quiet = l
  if (i .eq. 0) stop 3, quiet = .not. l
  if (i .eq. 0) stop 3, quiet = larray(1)
  if (i .eq. 0) stop , quiet = .false.
  if (i .eq. 0) error stop "Error."
  if (i .eq. 0) error stop chr1
!ERROR: CHARACTER stop code must be of default kind
  if (i .eq. 0) error stop chr2
  if (i .eq. 0) error stop 1
  if (i .eq. 0) error stop i
  if (i .eq. 0) error stop p_i
  if (i .eq. 0) error stop p_iarray(1)
  if (i .eq. 0) error stop iarray(1)
  if (i .eq. 0) error stop aiarray(1)
  if (i .eq. 0) error stop 1 + i
!ERROR: INTEGER stop code must be of default kind
  if (i .eq. 0) error stop invalid
!ERROR: Stop code must be of INTEGER or CHARACTER type
  if (i .eq. 0) error stop 12.34
  if (i .eq. 0) error stop 1, quiet = .true.
  if (i .eq. 0) error stop 2, quiet = .false.
  if (i .eq. 0) error stop 3, quiet = l
  if (i .eq. 0) error stop 3, quiet = .not. l
  if (i .eq. 0) error stop 3, quiet = larray(1)
  if (i .eq. 0) error stop , quiet = .false.
  stop
end program

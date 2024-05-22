! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
program main
  if (.true.) then
    do j = 1, 2
      goto 1 ! ok; used to cause looping in label resolution
    end do
  else
    goto 1 ! ok
1 end if
  if (.true.) then
    do j = 1, 2
      !WARNING: Label '1' is in a construct that should not be used as a branch target here
      goto 1
    end do
  end if
  !WARNING: Label '1' is in a construct that should not be used as a branch target here
  goto 1
end

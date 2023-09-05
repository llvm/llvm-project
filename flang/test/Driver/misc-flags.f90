! Make sure that `-l` is "visible" to Flang's driver
! RUN: %flang -lpgmath -### %s

! Make sure that `-Wl` is "visible" to Flang's driver
! RUN: %flang -Wl,abs -### %s

! Make sure that `-fuse-ld' is "visible" to Flang's driver
! RUN: %flang -fuse-ld= -### %s

! Make sure that `-L' is "visible" to Flang's driver
! RUN: %flang -L/ -### %s

program hello
  write(*,*), "Hello world!"
end program hello

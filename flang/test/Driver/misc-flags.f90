! Make sure that `-l` is "visible" to Flang's driver
! RUN: %flang -lpgmath -### %s

program hello
  write(*,*), "Hello world!"
end program hello

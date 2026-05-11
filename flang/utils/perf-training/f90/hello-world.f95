! RUN: %flang -c %s
! RUN: %flang_skip_driver -c %s

program hello_world
  implicit none

  print *, 'Hello, World!'
end program

!RUN: %flang -gdwarf-5 -S -emit-llvm %s -o - | FileCheck %s

!CHECK: !"Dwarf Version", i32 5

program main
  print *, "hello world !!"
end program main

! RUN: %flang -Qunused-arguments -c -o /dev/null %s -L. 2>&1 | FileCheck %s --allow-empty

! CHECK-NOT: argument unused during compilation

end program

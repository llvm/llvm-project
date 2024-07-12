! RUN: %flang -E %s 2>&1 | FileCheck --strict-whitespace %s
! CHECK:      print *, 666
pr&
&i&
&nt *, 666
end

! RUN: %flang -### -o /dev/null %s -Xlinker -rpath=/not/a/real/path 2>&1 | FileCheck %s

! CHECK: "-fc1"
! CHECK-NEXT: "-rpath=/not/a/real/path"

end program

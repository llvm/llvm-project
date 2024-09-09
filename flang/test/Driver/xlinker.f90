! RUN: %flang -### -o /dev/null %s -Xlinker -rpath=/not/a/real/path 2>&1 | FileCheck --check-prefix=SINGLE %s
! RUN: %flang -### -o /dev/null %s -Xlinker -rpath -Xlinker /not/a/real/path 2>&1 | FileCheck --check-prefix=MULTIPLE %s


! SINGLE: "-rpath=/not/a/real/path"
! MULTIPLE: "-rpath" "/not/a/real/path"

end program

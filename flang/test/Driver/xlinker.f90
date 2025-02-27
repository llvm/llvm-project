! RUN: %flang -### --target=ppc64le-linux-gnu -Xlinker -rpath -Xlinker /not/a/real/path %s 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=aarch64-apple-darwin -Xlinker -rpath -Xlinker /not/a/real/path %s 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=sparc-sun-solaris2.11 -Xlinker -rpath -Xlinker /not/a/real/path %s 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=x86_64-unknown-freebsd -Xlinker -rpath -Xlinker /not/a/real/path %s 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=x86_64-unknown-netbsd -Xlinker -rpath -Xlinker /not/a/real/path %s 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=x86_64-unknown-openbsd -Xlinker -rpath -Xlinker /not/a/real/path %s 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=x86_64-unknown-dragonfly -Xlinker -rpath -Xlinker /not/a/real/path %s 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=x86_64-unknown-haiku %s -Xlinker -rpath -Xlinker /not/a/real/path 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=x86_64-windows-gnu -Xlinker -rpath -Xlinker /not/a/real/path %s 2>&1 | FileCheck %s --check-prefixes=UNIX
! RUN: %flang -### --target=aarch64-windows-msvc -Xlinker -rpath -Xlinker /not/a/real/path -o obscure.exe %s 2>&1 | FileCheck %s --check-prefixes=MSVC

! UNIX-LABEL: "{{.*}}ld{{(\.exe)?}}"
! UNIX-SAME: "-rpath" "/not/a/real/path"

! The name of this file contains the word "link" which results in a match on
! the compiler line as well. Instead look for the final name of the executable
! to be created since that will only appear in the linker line.
! MSVC: -out:obscure.exe
! MSVC-SAME: "-rpath" "/not/a/real/path"

end program

! REQUIRES: classic_flang

! UNSUPPORTED: classic_flang

! -cpp should preprocess as it goes, regardless of input file extension
! RUN: %flang -cpp -c -DHELLO="hello all" -### %s 2>&1 | FileCheck %s --check-prefixes=ALL,CPP,PP
! RUN: %flang -cpp -c -DHELLO="hello all" -### -c f95-cpp-input %s 2>&1 | FileCheck %s --check-prefixes=ALL,CPP,PP
! -E should preprocess then stop, regardless of input file extension
! RUN: %flang -E -DHELLO="hello all" -### %s 2>&1 | FileCheck %s --check-prefixes=ALL,E,PPONLY
! RUN: %flang -E -DHELLO="hello all" -### -x f95-cpp-input %s 2>&1 | FileCheck %s --check-prefixes=ALL,E,PPONLY
! -cpp and -E are redundant
! RUN: %flang -E -cpp -DHELLO="hello all" -### %s 2>&1 | FileCheck %s --check-prefixes=ALL,E,PPONLY

! Don't link when given linker input
! RUN: %flang -E -cpp -Wl,-rpath=blah -### %s 2>&1 | FileCheck %s --check-prefixes=ALL,E,PPONLY

! Explicitly test this nonsence case causing a bug with LLVM 13/14
! RUN: %flang -E -traditional-cpp -DHELLO="hello all" -x f95-cpp-input -### %s 2>&1 | FileCheck %s --check-prefixes=ALL,E,PPONLY

! Test -save-temps does not break things (same codepath as -traditional-cpp bug above)
! RUN: %flang -E -DHELLO="hello all" -save-temps -### %s 2>&1 | FileCheck %s --check-prefixes=ALL,E,PPONLY
! RUN: %flang -E -DHELLO="hello all" -save-temps -### -c f95-cpp-input %s 2>&1 | FileCheck %s --check-prefixes=ALL,E,PPONLY
! RUN: %flang -cpp -c -DHELLO="hello all" -save-temps -### %s 2>&1 | FileCheck %s --check-prefixes=ALL,CPP
! RUN: %flang -cpp -c -DHELLO="hello all" -save-temps -### -c f95-cpp-input %s 2>&1 | FileCheck %s --check-prefixes=ALL,CPP

! Test for the correct cmdline flags
! Consume up to flang1 line
! ALL-LABEL: "{{.*}}flang1"
! CPP-NOT: "-es"
! CPP: "-preprocess"
! CPP-NOT: "-es"

! E-DAG: "-es"
! E-DAG: "-preprocess"

! flang1 should only be called once!
! ALL-NOT: "{{.*}}flang1"

! CPP should continue to build object
! PP: "{{.*}}flang2"
! PPONLY-NOT: "{{.*}}flang2"

! These commands should never call a linker!
! ALL-NOT: "{{.*}}ld"

program hello
  write(*, *) HELLO
end program hello


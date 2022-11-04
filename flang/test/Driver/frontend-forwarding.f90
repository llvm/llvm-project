! Test that flang-new forwards Flang frontend
! options to flang-new -fc1 as expected.

! RUN: %flang -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -finput-charset=utf-8 \
! RUN:     -fdefault-double-8 \
! RUN:     -fdefault-integer-8 \
! RUN:     -fdefault-real-8 \
! RUN:     -flarge-sizes \
! RUN:     -fconvert=little-endian \
! RUN:     -ffp-contract=fast \
! RUN:     -fno-honor-infinities \
! RUN:     -fno-honor-nans \
! RUN:     -fapprox-func \
! RUN:     -fno-signed-zeros \
! RUN:     -mllvm -print-before-all\
! RUN:     -P \
! RUN:   | FileCheck %s

! CHECK: "-P"
! CHECK: "-finput-charset=utf-8"
! CHECK: "-fdefault-double-8"
! CHECK: "-fdefault-integer-8"
! CHECK: "-fdefault-real-8"
! CHECK: "-flarge-sizes"
! CHECK: "-ffp-contract=fast"
! CHECK: "-menable-no-infs"
! CHECK: "-menable-no-nans"
! CHECK: "-fapprox-func"
! CHECK: "-fno-signed-zeros"
! CHECK: "-fconvert=little-endian"
! CHECK:  "-mllvm" "-print-before-all"

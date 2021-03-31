! Test that flang-new forwards Flang frontend
! options to flang-new -fc1 as expected.

! REQUIRES: new-flang-driver

! RUN: %flang-new -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -finput-charset=utf-8 \
! RUN:     -fdefault-double-8 \
! RUN:     -fdefault-integer-8 \
! RUN:     -fdefault-real-8 \
! RUN:     -flarge-sizes \
! RUN:   | FileCheck %s

! CHECK: "-finput-charset=utf-8"
! CHECK: "-fdefault-double-8"
! CHECK: "-fdefault-integer-8"
! CHECK: "-fdefault-real-8"
! CHECK: "-flarge-sizes"

! Test that flang-new forwards Flang frontend
! options to flang-new -fc1 as expected.

! REQUIRES: new-flang-driver

! RUN: %flang-new -fsyntax-only -### %s -o %t 2>&1 \
! RUN:     -finput-charset=utf-8 \
! RUN:   | FileCheck %s

! CHECK: "-finput-charset=utf-8"

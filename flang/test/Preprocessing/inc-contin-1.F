! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s
! CHECK: call t(1  ,.false.)
      program main
#include "inc-contin-1.h"
     $,.false.)
      end

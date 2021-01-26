! RUN: %f18 -E -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang-new -E -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang-new -fc1 -E -I %S/Inputs/ %s 2>&1 | FileCheck %s

! Test that the driver correctly reports diagnostics from the prescanner. The contents of the include file are irrelevant here.

! CHECK: preprocessor-diag.f90:8:20: #include: extra stuff ignored after file name
#include <empty.h> comment
! CHECK: preprocessor-diag.f90:10:20: #include: extra stuff ignored after file name
#include "empty.h" comment
end

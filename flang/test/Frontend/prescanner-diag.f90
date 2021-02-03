! Test that the driver correctly reports diagnostics from the prescanner. The contents of the include file are irrelevant here.

! Test with -E (i.e. PrintPreprocessedAction, stops after prescanning)
! RUN: %f18 -E -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang-new -E -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang-new -fc1 -E -I %S/Inputs/ %s 2>&1 | FileCheck %s

! Test with -fsyntax-only (i.e. ParseSyntaxOnlyAction, stops after semantic checks)
! RUN: %f18 -fsyntax-only -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang-new -fsyntax-only -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang-new -fc1 -fsyntax-only -I %S/Inputs/ %s 2>&1 | FileCheck %s

! CHECK: prescanner-diag.f90:14:20: #include: extra stuff ignored after file name
#include <empty.h> comment
! CHECK: prescanner-diag.f90:16:20: #include: extra stuff ignored after file name
#include "empty.h" comment
end

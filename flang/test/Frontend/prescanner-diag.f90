! Test that the driver correctly reports diagnostics from the prescanner. The contents of the include file are irrelevant here.

! Test with -E (i.e. PrintPreprocessedAction, stops after prescanning)
! RUN: %flang -E -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -E -I %S/Inputs/ %s 2>&1 | FileCheck %s

! Test with -fsyntax-only (i.e. ParseSyntaxOnlyAction, stops after semantic checks)
! RUN: %flang -fsyntax-only -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fsyntax-only -I %S/Inputs/ %s 2>&1 | FileCheck %s

! CHECK: prescanner-diag.f90:12:20: #include: extra stuff ignored after file name
#include <empty.h> comment
! CHECK: prescanner-diag.f90:14:20: #include: extra stuff ignored after file name
#include "empty.h" comment
end

// RUN: %clang -M -MG -include nonexistent-preinclude.h %s > %t
// RUN: fgrep nonexistent-preinclude.h %t
// RUN: fgrep nonexistent-ppinclude.h %t

#include "nonexistent-ppinclude.h"

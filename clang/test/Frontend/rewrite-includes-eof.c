// RUN: %clang_cc1 -E -frewrite-includes -I %S/Inputs %s
// expected-no-diagnostics
// Note: there's no newline at the end of this C file.
#include "rewrite-includes-bom.h"
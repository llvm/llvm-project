// RUN: %clang_cc1 -E -frewrite-includes %s | %clang_cc1 -
// expected-no-diagnostics
// Note: This source file has LF line endings.
// This test validates that -frewrite-includes translates the end of line (EOL)
// form used in header files to the EOL form used in the the primary source
// file when the files use different EOL forms.
#include "rewrite-includes-mixed-eol-crlf.h"
#include "rewrite-includes-mixed-eol-lf.h"

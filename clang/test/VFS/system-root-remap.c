// RUN: mkdir -p %t
// RUN: cd %t
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@EXTERNAL_DIR@%{/t:regex_replacement}@g" %S/Inputs/system-root-remap.yaml  > %t.yaml
// RUN: mkdir -p %t%t
// RUN: cp %S/Inputs/actual_header.h %t%t/not_real.h
// RUN: mkdir -p %t%S
// RUN: cp %s %t%S
// RUN: %clang_cc1 -Werror -I . -vfsoverlay %t.yaml -fsyntax-only -working-directory=%t %s

#include "not_real.h"

void foo(void) {
  bar();
}

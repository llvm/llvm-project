// RUN: mkdir -p %t
// RUN: cd %t
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -I . -ivfsoverlay %t.yaml -fsyntax-only %s

#include "not_real.h"

void foo(void) {
  bar();
}

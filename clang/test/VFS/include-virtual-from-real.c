// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo '#include "not_real.h"' > %t/include_not_real.h
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -ivfsoverlay %t.yaml -I %t -fsyntax-only %s

#include "include_not_real.h"

void foo(void) {
  bar();
}

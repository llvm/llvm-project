// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: echo "void baz(void);" > %t/real.h
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -ivfsoverlay %t.yaml -I %t -fsyntax-only %s

#include "not_real.h"
#include "real.h"

void foo() {
  bar();
  baz();
}

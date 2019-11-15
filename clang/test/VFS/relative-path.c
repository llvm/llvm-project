// RUN: mkdir -p %t
// RUN: cd %t
// RUN: sed -e "s@INPUT_DIR@%/S/Inputs@g" -e "s@OUT_DIR@%/t@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -I . -ivfsoverlay %t.yaml -fsyntax-only %s

#include "not_real.h"

void foo() {
  bar();
}

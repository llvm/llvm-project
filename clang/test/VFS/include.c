// RUN: sed -e "s@INPUT_DIR@%/S/Inputs@g" -e "s@OUT_DIR@%/t@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -I %t -ivfsoverlay %t.yaml -fsyntax-only %s

// FIXME: PR43272
// XFAIL: system-windows

#include "not_real.h"

void foo() {
  bar();
}

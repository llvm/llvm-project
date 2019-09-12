// RUN: sed -e "s@INPUT_DIR@%/S/Inputs@g" -e "s@OUT_DIR@%/t@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -ivfsoverlay %t.yaml -I %t -include "not_real.h" -fsyntax-only %s

// FIXME: PR43272
// XFAIL: system-windows

void foo() {
  bar();
}

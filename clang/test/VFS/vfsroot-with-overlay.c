// REQUIRES: shell
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: sed -e "s:TEST_DIR:%S:g" -e "s:OUT_DIR:%t:g" %S/Inputs/vfsroot.yaml > %t.yaml
// RUN: sed -e "s:INPUT_DIR:/indirect-vfs-root-files:g" -e "s:OUT_DIR:/overlay-dir:g" %S/Inputs/vfsoverlay.yaml > %t/vfsoverlay.yaml
// RUN: %clang_cc1 -Werror -ivfsoverlay %t.yaml -ivfsoverlay /direct-vfs-root-files/vfsoverlay.yaml -I /overlay-dir -fsyntax-only /tests/vfsroot-with-overlay.c

#include "not_real.h"

void foo() {
  bar();
}

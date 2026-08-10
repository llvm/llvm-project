// RUN: mkdir -p %t
// RUN: cd %t
// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -I . -ivfsoverlay %t.yaml -fsyntax-only %s

// RUN: cp %S/Inputs/actual_header.h %t/actual_header.h
// RUN: sed -e "s@OVERLAY_DIR@true@g" -e "s@EXTERNAL_DIR@.@g" %S/Inputs/root-relative-overlay.yaml > %t/root-relative-overlay.yaml
// RUN: %clang_cc1 -Werror -I %t -ivfsoverlay %t/root-relative-overlay.yaml -fsyntax-only %s

// RUN: sed -e "s@OVERLAY_DIR@false@g" -e "s@EXTERNAL_DIR@%{/t:regex_replacement}@g" %S/Inputs/root-relative-overlay.yaml > %t/root-relative-overlay2.yaml
// RUN: %clang_cc1 -Werror -I %t -ivfsoverlay %t/root-relative-overlay2.yaml -fsyntax-only %s

#include "not_real.h"

void foo(void) {
  bar();
}

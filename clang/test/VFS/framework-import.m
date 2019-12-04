// RUN: sed -e "s@INPUT_DIR@%{/S:regex_replacement}/Inputs@g" -e "s@OUT_DIR@%{/t:regex_replacement}@g" %S/Inputs/vfsoverlay.yaml > %t.yaml
// RUN: %clang_cc1 -Werror -F %t -ivfsoverlay %t.yaml -fsyntax-only %s

#import <SomeFramework/public_header.h>

void foo() {
  from_framework();
}

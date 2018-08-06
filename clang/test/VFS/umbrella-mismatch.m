// RUN: rm -rf %t
// RUN: sed -e "s;INPUT_DIR;%/S/Inputs;g" -e "s;OUT_DIR;%/S/Inputs;g" %S/Inputs/vfsoverlay.yaml > %t.yaml

// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -ivfsoverlay %t.yaml -F %S/Inputs -fsyntax-only %s -Wno-atimport-in-framework-header -verify
// RUN: %clang_cc1 -Werror -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs -fsyntax-only %s -Wno-atimport-in-framework-header -verify
// expected-no-diagnostics
@import UsesFoo;

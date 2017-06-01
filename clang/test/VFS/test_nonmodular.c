// REQUIRES: shell

// RUN: rm -rf %t
// RUN: mkdir -p %t/vdir %t/cache %t/outdir
// We can't have module.map inside Inputs/Nonmodular.
// RUN: cp %S/Inputs/Nonmodular/Nonmodular.modulemap %t/outdir/module.modulemap
//
// RUN: sed -e "s:VDIR:%t/vdir:g" -e "s:IN_DIR:%S:g" -e "s:OUT_DIR:%t/outdir:g" %S/Inputs/Nonmodular/nonmodular-headers.yaml > %t/vdir/nonmodular-headers.yaml
// RUN: %clang_cc1 -fmodule-name=Nonmodular -fmodules -Wnon-modular-include-in-framework-module -verify -fimplicit-module-maps -fmodules-cache-path=%t/cache -ivfsoverlay %t/vdir/nonmodular-headers.yaml -I %S/Inputs -F %t/vdir -fsyntax-only %S/Inputs/Nonmodular/test.c

// expected-no-diagnostics

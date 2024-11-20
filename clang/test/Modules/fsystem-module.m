// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir

// -fsystem-module requires -emit-module
// RUN: not %clang_cc1 -fsyntax-only -fsystem-module %s 2>&1 | grep "-emit-module"

// RUN: not %clang_cc1 -fmodules -I %S/Inputs \
// RUN:   -emit-module -fmodule-name=warning -pedantic -Werror \
// RUN:   %S/Inputs/module.modulemap -o %t.dir/warning.pcm

// RUN: %clang_cc1 -fmodules -I %S/Inputs \
// RUN:   -emit-module -fmodule-name=warning -pedantic -Werror \
// RUN:   %S/Inputs/module.modulemap -o %t.dir/warning-system.pcm -fsystem-module

// RUN: not %clang_cc1 -fmodules -I %S/Inputs \
// RUN:   -emit-module -fmodule-name=warning -pedantic -Werror \
// RUN:   %S/Inputs/module.modulemap -o %t.dir/warning-system.pcm -fsystem-module \
// RUN:   -Wsystem-headers

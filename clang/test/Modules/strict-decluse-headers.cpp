// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: touch %t/foo.h
// RUN: echo '#include "foo.h"' > %t/bar.h
// RUN: touch %t/baz.h
// RUN: echo 'module X { header "bar.h" header "baz.h" }' > %t/map
//
// RUN: not %clang_cc1 -fsyntax-only -fmodules -fmodule-map-file=%t/map -I%t -fmodules-strict-decluse -fmodule-name=X -x c++ %t/bar.h %t/baz.h 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fmodules -fmodule-map-file=%t/map -I%t -fmodules-strict-decluse -fmodule-name=X -x c++ %t/baz.h %t/bar.h 2>&1 | FileCheck %s
//
// Don't crash on this: (FIXME: we should produce an error that the specified module name is not known)
// RUN: %clang_cc1 -fsyntax-only -fmodules -I%t -fmodules-strict-decluse -fmodule-name=X -x c++ %t/baz.h %t/bar.h
//
// Don't crash on this: (FIXME: we should produce an error that the specified file is not part of the specified module)
// RUN: %clang_cc1 -fsyntax-only -fmodules -fmodule-map-file=%t/map -I%t -fmodules-strict-decluse -fmodule-name=X -x c++ %t/foo.h
//
// CHECK: module X does not depend on a module exporting 'foo.h'

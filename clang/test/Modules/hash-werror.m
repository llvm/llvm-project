// RUN: rm -rf %t
// RUN: mkdir %t

// (1) Test -Werror
// RUN: echo 'int foo() { return fn(); }' > %t/foo.h
// RUN: echo 'module foo { header "foo.h" export * }' > %t/module.modulemap
//
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps %s -fsyntax-only \
// RUN:   -I%t -fmodules-cache-path=%t/cache -Rmodule-build \
// RUN:   -fmodules-hash-error-diagnostics 2>%t/out
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps %s -fsyntax-only \
// RUN:   -I%t -Werror -fmodules-cache-path=%t/cache -Rmodule-build \
// RUN:   -Wno-implicit-function-declaration \
// RUN:   -fmodules-hash-error-diagnostics 2>>%t/out

// RUN: FileCheck --check-prefix=CHECKWERROR %s -input-file %t/out
// CHECKWERROR: remark: building module 'foo' as '[[PATH:.*]]{{/|\\}}foo-
// CHECKWERROR-NOT: remark: building module 'foo' as '[[PATH]]{{/|\\}}foo-

// (2) Test -Werror=
// RUN: rm -rf %t/out %t/cache
//
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps %s -fsyntax-only \
// RUN:   -I%t -Werror -fmodules-cache-path=%t/cache -Rmodule-build \
// RUN:   -fmodules-hash-error-diagnostics 2>%t/out
//
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps %s -fsyntax-only \
// RUN:   -I%t -fmodules-cache-path=%t/cache -Rmodule-build \
// RUN:   -Werror=implicit-function-declaration \
// RUN:   -fmodules-hash-error-diagnostics 2>>%t/out

// RUN: FileCheck --check-prefix=CHECKWERROREQUALS %s -input-file %t/out
// CHECKWERROREQUALS: remark: building module 'foo' as '[[PATH:.*]]{{/|\\}}foo-
// CHECKWERROREQUALS-NOT: remark: building module 'foo' as '[[PATH]]{{/|\\}}foo-

// (3) Test -pedantic-errors
// RUN: rm -rf %t/out %t/cache
// RUN: echo '#ifdef foo' > %t/foo.h
// RUN: echo '#endif bad // extension!' >> %t/foo.h

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps %s -fsyntax-only \
// RUN:   -I%t -fmodules-cache-path=%t/cache -Rmodule-build -x c \
// RUN:   -fmodules-hash-error-diagnostics 2>%t/out
//
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps %s -fsyntax-only \
// RUN:   -I%t -pedantic-errors -fmodules-cache-path=%t/cache -Rmodule-build -x c \
// RUN:   -fmodules-hash-error-diagnostics 2>>%t/out

// RUN: FileCheck --check-prefix=CHECKPEDANTICERROR %s -input-file %t/out
// CHECKPEDANTICERROR: remark: building module 'foo' as '[[PATH:.*]]{{/|\\}}foo-
// CHECKPEDANTICERROR-NOT: remark: building module 'foo' as '[[PATH]]{{/|\\}}foo-

#include "foo.h"

// Test scanning deps does not have more errors than the regular compilation.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// Check the regular compilation does not fail.
// RUN: %clang -fsyntax-only %t/test.c -I %t/include -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -Wnon-modular-include-in-module -Werror=non-modular-include-in-module

// And now scanning deps should succeed too.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/test.c -I DIR/include -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -Wnon-modular-include-in-module -Werror=non-modular-include-in-module",
    "file": "DIR/test.c"
  },
]

//--- include/nonmodular.h
// empty

//--- include/modular-includer.h
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-modular-include-in-module"
#include <nonmodular.h>
#pragma clang diagnostic pop

//--- include/module.modulemap
module ModularIncluder { header "modular-includer.h" }

//--- test.c
#include <modular-includer.h>

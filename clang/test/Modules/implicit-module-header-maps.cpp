// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %hmaptool write a.hmap.json hmap
//
// RUN: %clang -Rmodule-build -fmodules -fimplicit-modules -fimplicit-module-maps -fmodule-map-file=module.modulemap -fsyntax-only -I hmap -fmodules-cache-path=%t test.cpp
//
// RUN: cd %T
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: sed -e "s|OUTPUTS_DIR|%t|g" b.hmap.json > hmap.json
// RUN: %hmaptool write hmap.json hmap
//
// RUN: %clang -Rmodule-build -fmodules -fimplicit-modules -fimplicit-module-maps -fmodule-map-file=module.modulemap -fsyntax-only -I hmap -fmodules-cache-path=%t test.cpp

//--- After/Mapping.h
#ifdef FOO
#error foo
#endif

//--- a.hmap.json
{
  "mappings" :
    {
     "Before/Mapping.h" : "After/Mapping.h",
     "After/Mapping.h" : "After/Mapping.h"
    }
}

//--- b.hmap.json
{
  "mappings" :
    {
     "Before/Mapping.h" : "OUTPUTS_DIR/After/Mapping.h"
    }
}

//--- module.modulemap
module a {
  header "After/Mapping.h"
}


//--- test.cpp
#define FOO
// This include will fail if:
// 1) modules are't used, as the `FOO` define will propagate into the included
//    header and trip a `#error`, or
// 2) header maps aren't used, as the header name doesn't exist and relies on
//    the header map to remap it to the real header.
#include "Before/Mapping.h"

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -fmodules-lazy-load-module-maps -verify

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t \
// RUN:   -fmodules-cache-path=%t/cache %t/tu.c -fsyntax-only -Rmodule-map \
// RUN:   -DEAGER -verify

// Test that extern module declarations are followed when building the header
// cache, so headers declared in external module maps can be found. This tests
// regular headers, umbrella headers, and umbrella directories in extern
// module maps.

//--- module.modulemap

extern module Ext "sub/extern.modulemap"
extern module ExtUmbrellaHeader "umbrella_hdr/extern.modulemap"
extern module ExtUmbrellaDir "umbrella_dir/extern.modulemap"

//--- sub/extern.modulemap

// Regular header in extern module.
module Ext {
  header "ext.h"
}

//--- umbrella_hdr/extern.modulemap

module ExtUmbrellaHeader {
  umbrella header "umbrella.h"
}

//--- umbrella_dir/extern.modulemap

module ExtUmbrellaDir {
  umbrella "inc"
}

//--- sub/ext.h

//--- umbrella_hdr/umbrella.h
#include "covered.h"

//--- umbrella_hdr/covered.h

//--- umbrella_dir/inc/from_umbrella.h

//--- tu.c

#include <sub/ext.h>
#include <umbrella_hdr/covered.h>
#include <umbrella_dir/inc/from_umbrella.h>

#ifndef EAGER
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{parsing modulemap}}
// expected-remark@*{{loading parsed module 'Ext'}}
// expected-remark@*{{loading parsed module 'ExtUmbrellaHeader'}}
// expected-remark@*{{loading parsed module 'ExtUmbrellaDir'}}
#else
// expected-remark@*{{loading modulemap}}
// expected-remark@*{{loading modulemap}}
// expected-remark@*{{loading modulemap}}
// expected-remark@*{{loading modulemap}}
#endif

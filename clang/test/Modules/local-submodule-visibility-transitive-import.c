// Checks that macros from transitive imports work with local submodule
// visibility. In the below test, previously a() and d() failed because
// OTHER_MACRO1 and OTHER_MACRO3 were not visible at the use site.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -fmodules-local-submodule-visibility -I%t %t/tu.c -verify

//--- Other1.h
#define OTHER_MACRO1(...)

//--- Other2.h
#define OTHER_MACRO2(...)

//--- Other3.h
#define OTHER_MACRO3(...)

//--- module.modulemap
module Other {
  module O1 { header "Other1.h" }
  module O2 { header "Other2.h" }
  module O3 { header "Other3.h" }
}

//--- Top/A.h
#include "Other1.h"
#define MACRO_A OTHER_MACRO1(x, y)

//--- Top/B.h
#include "Other2.h"
#define MACRO_B OTHER_MACRO2(x, y)

//--- Top/C.h
#include "D.h"

//--- Top/D.h
#include "Other3.h"
#define MACRO_D OTHER_MACRO3(x, y)

//--- Top/Top.h
#include "A.h"
#include "B.h"
#include "C.h"

void a() MACRO_A;
void b() MACRO_B;
void d() MACRO_D;

//--- Top/module.modulemap
module Top {
  umbrella header "Top.h"
  module A { header "A.h" export * }
  module D { header "D.h" export * }
  module * { export * }
  export *
  export Other.O3
}

//--- tu.c
#include "Top/Top.h"
// expected-no-diagnostics

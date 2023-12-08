// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs/suggest-include %s -verify

#include "empty.h" // import the module file

// expected-note@usetextual1.h:2 {{here}}
// expected-note@textual2.h:1 {{here}}
// expected-note@textual3.h:1 {{here}}
// expected-note@textual4.h:1 {{here}}
// expected-note@textual5.h:1 {{here}}
// expected-note@private1.h:1 {{here}}
// expected-note@private2.h:1 {{here}}
// expected-note@private3.h:1 {{here}}

void f() {
  (void)::usetextual1; // expected-error {{missing '#include "usetextual1.h"'}}
  (void)::usetextual2; // expected-error {{missing '#include "usetextual2.h"'}}
  (void)::textual3; // expected-error-re {{{{^}}missing '#include "usetextual3.h"'}}
  // If the declaration is in an include-guarded header, make sure we suggest
  // including that rather than importing a module. In this case, there could
  // be more than one module, and the module name we picked is almost certainly
  // wrong.
  (void)::textual4; // expected-error {{missing '#include "usetextual4.h"'; 'textual4' must be declared before it is used}}
  (void)::textual5; // expected-error {{missing '#include "usetextual5.h"'; 'textual5' must be declared before it is used}}

  // Don't suggest #including a private header.
  // FIXME: We could suggest including "useprivate1.h" here, as it's the only
  // public way to get at this declaration.
  (void)::private1; // expected-error-re {{{{^}}declaration of 'private1'}}
  // FIXME: Should we be suggesting an import at all here? Should declarations
  // in private headers be visible when the surrounding module is imported?
  (void)::private2; // expected-error-re {{{{^}}declaration of 'private2'}}
  // Even if we suggest an include for private1, we should not do so here.
  (void)::private3; // expected-error-re {{{{^}}declaration of 'private3'}}
}

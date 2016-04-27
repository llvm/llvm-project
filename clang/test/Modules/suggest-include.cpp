// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs/suggest-include %s -verify

#include "empty.h" // import the module file

// expected-note@usetextual1.h:2 {{previous}}
// expected-note@textual2.h:1 {{previous}}
// expected-note@textual3.h:1 {{previous}}
// expected-note@textual4.h:1 {{previous}}
// expected-note@textual5.h:1 {{previous}}
// expected-note@private1.h:1 {{previous}}
// expected-note@private2.h:1 {{previous}}
// expected-note@private3.h:1 {{previous}}

void f() {
  (void)::usetextual1; // expected-error {{missing '#include "usetextual1.h"'}}
  (void)::usetextual2; // expected-error {{missing '#include "usetextual2.h"'}}
  (void)::textual3; // expected-error-re {{{{^}}missing '#include "usetextual3.h"'}}
  // Don't suggest a #include that includes the entity via a path that leaves
  // the module. In that case we can't be sure that we've picked the right header.
  (void)::textual4; // expected-error-re {{{{^}}declaration of 'textual4'}}
  (void)::textual5; // expected-error-re {{{{^}}declaration of 'textual5'}}

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

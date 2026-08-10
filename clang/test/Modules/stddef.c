// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fbuiltin-headers-in-system-modules -fmodules-cache-path=%t -I%S/Inputs/StdDef %s -verify=builtin-headers-in-system-modules -fno-modules-error-recovery
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/StdDef %s -verify=no-builtin-headers-in-system-modules -fno-modules-error-recovery

#include "ptrdiff_t.h"

ptrdiff_t pdt;

// size_t is declared in both size_t.h and __stddef_size_t.h. If
// -fbuiltin-headers-in-system-modules is set, then __stddef_size_t.h is a
// non-modular header that will be transitively pulled in the StdDef test module
// by include_again.h. Otherwise it will be in the _Builtin_stddef module. In
// any case it's not defined which module will win as the expected provider of
// size_t. For the purposes of this test it doesn't matter which of the two
// providing headers get reported.
size_t st; // builtin-headers-in-system-modules-error-re {{missing '#include "{{size_t|include_again}}.h"'; 'size_t' must be declared before it is used}} \
              no-builtin-headers-in-system-modules-error-re {{missing '#include "{{size_t|__stddef_size_t}}.h"'; 'size_t' must be declared before it is used}}
// builtin-headers-in-system-modules-note@size_t.h:* 0+ {{here}} \
   no-builtin-headers-in-system-modules-note@size_t.h:* 0+ {{here}}
// builtin-headers-in-system-modules-note@__stddef_size_t.h:* 0+ {{here}} \
   no-builtin-headers-in-system-modules-note@__stddef_size_t.h:* 0+ {{here}}

#include "include_again.h"
// Includes <stddef.h> which includes <__stddef_size_t.h>.

size_t st2;

#include "size_t.h"
// Redeclares size_t when -fbuiltin-headers-in-system-modules is not passed, but
// the type merger should figure it out.

size_t st3;

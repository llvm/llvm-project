// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fbuiltin-headers-in-system-modules -fmodules-cache-path=%t -I%S/Inputs/StdDef %s -verify -fno-modules-error-recovery
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/StdDef %s -verify -fno-modules-error-recovery

#include "ptrdiff_t.h"

ptrdiff_t pdt;

// size_t is declared in both size_t.h and __stddef_size_t.h, both of which are
// modular headers. Regardless of whether stddef.h joins the StdDef test module
// or is in its _Builtin_stddef module, __stddef_size_t.h will be in
// _Builtin_stddef.size_t. It's not defined which module will win as the expected
// provider of size_t. For the purposes of this test it doesn't matter which header
// gets reported, just as long as it isn't other.h or include_again.h.
size_t st; // expected-error-re {{missing '#include "{{size_t|__stddef_size_t}}.h"'; 'size_t' must be declared before it is used}}
// expected-note@size_t.h:* 0+ {{here}}
// expected-note@__stddef_size_t.h:* 0+ {{here}}

#include "include_again.h"
// Includes <stddef.h> which includes <__stddef_size_t.h> which imports the
// _Builtin_stddef.size_t module.

size_t st2;

#include "size_t.h"
// Redeclares size_t, but the type merger should figure it out.

size_t st3;

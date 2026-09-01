// This test checks that -Wshadow-header reports each shadowed header only
// once, no matter how many times it is looked up.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -Eonly %t/tu.c -I %t/include1 -I %t/include2 \
// RUN:   -Wshadow-header -verify

//--- tu.c
// expected-warning-re@+1 {{multiple candidates for header 'shadowed1.h' found; directory '{{.*}}include1' chosen, ignoring others including '{{.*}}include2'}}
#include "shadowed1.h"
// Including the same header again must not report the shadowing again.
#include "shadowed1.h"
// Neither must looking it up from a different includer.
#include "sub/reinclude.h"
// A different shadowed header is still reported.
// expected-warning-re@+1 {{multiple candidates for header 'shadowed2.h' found; directory '{{.*}}include1' chosen, ignoring others including '{{.*}}include2'}}
#include "shadowed2.h"
// The candidates depend on the spelling of the include, so the same file is
// still reported once per spelling it is found under: this one is shadowed by
// 'include2/a/spelled.h', the one in 'include1/a/reinclude.h' by
// 'include2/spelled.h'.
// expected-warning-re@+1 {{multiple candidates for header 'a/spelled.h' found; directory '{{.*}}include1{{.*}}a' chosen, ignoring others including '{{.*}}include2'}}
#include "a/spelled.h"
#include "a/reinclude.h"

//--- sub/reinclude.h
#include "shadowed1.h"

//--- include1/shadowed1.h
//--- include2/shadowed1.h
//--- include1/shadowed2.h
//--- include2/shadowed2.h
//--- include1/a/spelled.h
//--- include2/a/spelled.h
//--- include2/spelled.h
//--- include1/a/reinclude.h
// expected-warning-re@+1 {{multiple candidates for header 'spelled.h' found; directory '{{.*}}include1{{.*}}a' chosen, ignoring others including '{{.*}}include2'}}
#include "spelled.h"

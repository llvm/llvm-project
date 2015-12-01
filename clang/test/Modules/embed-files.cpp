// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'module a { header "a.h" } module b { header "b.h" }' > %t/modulemap
// RUN: echo 'extern int t;' > %t/t.h
// RUN: echo '#include "t.h"' > %t/a.h
// RUN: echo '#include "t.h"' > %t/b.h

// RUN: %clang_cc1 -fmodules -I%t -fmodules-cache-path=%t -fmodule-map-file=%t/modulemap -fmodules-embed-all-files %s -verify
#include "a.h"
char t; // expected-error {{different type}}
// expected-note@t.h:1 {{here}}
#include "t.h"
#include "b.h"
char t; // expected-error {{different type}}
// expected-note@t.h:1 {{here}}

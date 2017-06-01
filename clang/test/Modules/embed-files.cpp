// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'module a { header "a.h" header "x.h" } module b { header "b.h" }' > %t/modulemap
// RUN: echo 'extern int t;' > %t/t.h
// RUN: echo '#include "t.h"' > %t/a.h
// RUN: echo '#include "t.h"' > %t/b.h
// RUN: echo '#include "t.h"' > %t/x.h

// RUN: %clang_cc1 -fmodules -I%t -fmodules-cache-path=%t -fmodule-map-file=%t/modulemap -fmodules-embed-all-files %s -verify
//
// RUN: %clang_cc1 -fmodules -I%t -fmodules-embed-all-files %t/modulemap -fmodule-name=a -x c++ -emit-module -o %t/a.pcm
// RUN: %clang_cc1 -fmodules -I%t -fmodules-embed-all-files %t/modulemap -fmodule-name=b -x c++ -emit-module -o %t/b.pcm
// FIXME: This test is flaky on Windows because attempting to delete a file
// after writing it just doesn't seem to work well, at least not in the lit
// shell.
// REQUIRES: shell
// RUN: rm %t/x.h
// RUN: %clang_cc1 -fmodules -I%t -fmodule-map-file=%t/modulemap -fmodule-file=%t/a.pcm -fmodule-file=%t/b.pcm %s -verify
#include "a.h"
char t; // expected-error {{different type}}
// expected-note@t.h:1 {{here}}
#include "t.h"
#include "b.h"
char t; // expected-error {{different type}}
// expected-note@t.h:1 {{here}}


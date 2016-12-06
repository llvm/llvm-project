// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'module M { header "foo.h" header "bar.h" }' > %t/map
// RUN: echo 'template<typename T> void f(T t) { int n; t.f(n); }' > %t/foo.h
// RUN: touch %t/bar.h
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -x c++ %t/map -emit-module -fmodule-name=M -o %t/pcm
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodule-map-file=%t/map -fmodule-file=%t/pcm -I%t %s -fsyntax-only -fexceptions -Wno-module-file-config-mismatch -verify
// RUN: rm %t/bar.h
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodule-map-file=%t/map -fmodule-file=%t/pcm -I%t %s -fsyntax-only -fexceptions -Wno-module-file-config-mismatch -verify
#include "foo.h"
namespace n { // expected-note {{begins here}}
#include "foo.h" // expected-error {{redundant #include of module 'M' appears within namespace}}
}

// This tests that the compiler wouldn't crash if the module path misses

// RUN: rm -rf %t
// RUN: mkdir -p %t/subdir
// RUN: echo "export module C;" >> %t/subdir/C.cppm
// RUN: echo -e "export module B;\nimport C;" >> %t/B.cppm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/subdir/C.cppm -o %t/subdir/C.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fprebuilt-module-path=%t/subdir %t/B.cppm -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=B=%t/B.pcm %s -fsyntax-only -verify

import B; // expected-error {{failed to find module file for module 'C'}}
import C; // expected-error {{module 'C' not found}}

// This tests that the compiler wouldn't crash if the module path misses

// RUN: rm -rf %t
// RUN: mkdir -p %t/subdir
// RUN: echo "export module C;" >> %t/subdir/C.cppm
// RUN: echo -e "export module B;\nimport C;" >> %t/B.cppm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/subdir/C.cppm -o %t/subdir/C.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fprebuilt-module-path=%t/subdir %t/B.cppm -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %s -fsyntax-only -verify

import B;
import C; // expected-error {{module 'C' is needed but has not been provided, and implicit use of module files is disabled}}

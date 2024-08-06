// RUN: %clang_cl /std:c++20 --precompile -### -- %s 2>&1 | FileCheck  %s
// CHECK: --precompile

// RUN: %clang_cl /std:c++20 --fmodule-file=Foo=Foo.pcm -### -- %s 2>&1 | FileCheck  %s
// CHECK: -fmodule-file=Foo=Foo.pcm

// RUN: %clang_cl /std:c++20 --fprebuilt-module-path=. -### -- %s 2>&1 | FileCheck  %s
// CHECK: -fprebuilt-module-path=.
// RUN: %clang_cl /std:c++20 --precompile -### -- %s 2>&1 | FileCheck --check-prefix=PRECOMPILE %s
// PRECOMPILE: -emit-module-interface

// RUN: %clang_cl /std:c++20 --fmodule-file=Foo=Foo.pcm -### -- %s 2>&1 | FileCheck --check-prefix=FMODULEFILE %s
// FMODULEFILE: -fmodule-file=Foo=Foo.pcm

// RUN: %clang_cl /std:c++20 --fprebuilt-module-path=. -### -- %s 2>&1 | FileCheck --check-prefix=FPREBUILT %s
// FPREBUILT: -fprebuilt-module-path=.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cl /std:c++20 --precompile -### -- %s 2>&1 | FileCheck --check-prefix=PRECOMPILE %s
// PRECOMPILE: -emit-module-interface

// RUN: %clang_cl /std:c++20 --fmodule-file=Foo=Foo.pcm -### -- %s 2>&1 | FileCheck --check-prefix=FMODULEFILE %s
// FMODULEFILE: -fmodule-file=Foo=Foo.pcm

// RUN: %clang_cl /std:c++20 --fprebuilt-module-path=. -### -- %s 2>&1 | FileCheck --check-prefix=FPREBUILT %s
// FPREBUILT: -fprebuilt-module-path=.

// RUN: %clang_cl %t/test.pcm /std:c++20 -### 2>&1 | FileCheck --check-prefix=CPP20WARNING %t/test.pcm

//--- test.pcm
// CPP20WARNING-NOT: clang-cl: warning: argument unused during compilation: '/std:c++20' [-Wunused-command-line-argument]

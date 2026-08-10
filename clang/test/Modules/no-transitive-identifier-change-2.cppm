// Test that adding a new identifier within reduced BMI may not produce a transitive change.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/A.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/B.cppm -o %t/B.pcm \
// RUN:     -fmodule-file=A=%t/A.pcm
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/A.v1.cppm -o %t/A.v1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/B.cppm -o %t/B.v1.pcm \
// RUN:     -fmodule-file=A=%t/A.v1.pcm
//
// RUN: diff %t/B.pcm %t/B.v1.pcm &> /dev/null

//--- A.cppm
export module A;
export int a();

//--- A.v1.cppm
export module A;
export int a();
export int a2();

//--- B.cppm
export module B;
import A;
export int b() { return a(); }

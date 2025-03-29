// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface a.cppm -o a.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface b.cppm -o b.pcm \
// RUN:   -fmodule-file=A=a.pcm

// This test addresses issue #132059:
// Bad use of fmodule-file=<name>=<path/to/bmi> previously caused the compiler
// to crash for the following cases:

//--- a.cppm
export module A;

export int a() {
  return 41;
}

//--- b.cppm
export module B;
import A;

export int b() {
  return a() + 1;
}

// Test that when -fmodule-file=<name>=<path/to/bmi> mistakenly maps a module to
// a BMI which depends on the module, the compiler doesn't crash.

// RUN: not %clang_cc1 -std=c++20 main1.cpp-fmodule-file=B=b.pcm \
// RUN:   -fmodule-file=A=b.pcm

//--- main1.cpp
import B;

int main() {
  return b();
}

// Test that when -fmodule-file=<name>=<path/to/bmi> mistakenly maps a module
// to a BMI file, and that BMI exposes the parts of the specified module as
// transitive imports, the compiler doesn't crash.

// RUN: not %clang_cc1 -std=c++20 main2.cpp -fmodule-file=A=b.pcm

//--- main2.cpp
import A;

int main() {
  return a();
}

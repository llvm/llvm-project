// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// Related to issue #132059

// Precompile the module dependencies correctly 
// RUN: %clang_cc1 -std=c++20 -emit-module-interface a.cppm -o a.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface b.cppm -o b.pcm \
// RUN:     -fmodule-file=A=a.pcm

// Verify that providing incorrect mappings via
// `-fmodule-file=<name>=<path/to/bmi>` does not crash the compiler when loading
// a module that imports the incorrectly mapped module.
// RUN: not %clang_cc1 -std=c++20 main1.cpp -fmodule-file=A=b.pcm

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

//--- main1.cpp
import A;

int main() {
  return a();
}

// Test again for the case where the BMI is first loaded correctly
// RUN: not %clang_cc1 -std=c++20 main2.cpp-fmodule-file=B=b.pcm \
// RUN:     -fmodule-file=A=b.pcm

//--- main2.cpp
import B;

int main() {
  return b();
}

// Tests for -fkeep-inline-functions behaviour with C++20 named modules.
//
// Exported inline definitions are owned by the module interface unit and are
// retained there. Imported inline definitions are only affected if their
// definition is available in the current TU. If no use requires deserializing
// the definition, the function is not visible to this option.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// Compile the module interface unit to verify that the exported inline
// definition is owned by this TU and retained.
// RUN: %clang_cc1 -std=c++20 -fkeep-inline-functions -O2 \
// RUN:   %t/Hello.cppm -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-MODULE

// Build the PCM for the import tests below.
// RUN: %clang_cc1 -std=c++20 -fkeep-inline-functions -emit-module-interface \
// RUN:   %t/Hello.cppm -o %t/Hello.pcm

// Compile a TU that imports but does not call hello().
// The definition is not deserialized, so it is not retained.
// RUN: %clang_cc1 -std=c++20 -fkeep-inline-functions -O2 \
// RUN:   -fmodule-file=Hello=%t/Hello.pcm %t/no-use.cpp -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-USE

// Compile a TU that imports and calls hello().
// Calling hello() deserializes its definition, allowing
// -fkeep-inline-functions to retain it.
// RUN: %clang_cc1 -std=c++20 -fkeep-inline-functions -O2 \
// RUN:   -fmodule-file=Hello=%t/Hello.pcm %t/use.cpp -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-USE

//--- Hello.cppm
export module Hello;

// CHECK-MODULE: @llvm{{(\.compiler)?}}.used = {{.*}}hello
// CHECK-MODULE: define {{.*}}@_ZW5Hello5hellov
export inline int hello() { return 55; }

//--- use.cpp
import Hello;

// CHECK-USE: @llvm{{(\.compiler)?}}.used = {{.*}}hello
// CHECK-USE: define {{.*}}@_ZW5Hello5hellov
int main() {
  return hello();
}

//--- no-use.cpp
import Hello;

// CHECK-NO-USE-NOT: @llvm{{(\.compiler)?}}.used = {{.*}}hello
// CHECK-NO-USE-NOT: define {{.*}}@_ZW5Hello5hellov
int main() {
  return 0;
}


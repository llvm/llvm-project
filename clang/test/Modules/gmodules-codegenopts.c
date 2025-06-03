// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// Check that the output from -gmodules can be loaded back by the compiler in
// the presence of certain options like optimization level that could break
// output. Note: without compiling twice the module is loaded from the in-memory
// module cache not load it from the object container.

// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodules -fmodule-format=obj \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t %s \
// RUN:   -I %S/Inputs -verify -O2

// Compile again, confirming we can load the module.
// RUN: %clang_cc1 -x objective-c -fmodules -fmodule-format=obj \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t %s \
// RUN:   -I %S/Inputs -verify -O2

@import DebugObjC;
// expected-no-diagnostics
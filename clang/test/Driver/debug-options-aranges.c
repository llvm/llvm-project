// REQUIRES: lld

/// Check that the linker plugin will get -generate-arange-section.
// RUN: %clang -### -g --target=x86_64-linux -flto      -gdwarf-aranges %s 2>&1 | FileCheck %s
// RUN: %clang -### -g --target=x86_64-linux -flto=thin -gdwarf-aranges %s 2>&1 | FileCheck %s
// CHECK: "-plugin-opt=-generate-arange-section"

// RUN: %clang -fplugin=%llvmshlibdir/MicrosoftAttributes%pluginext -fms-extensions -E %s | FileCheck %s --check-prefix=MS
// REQUIRE: plugins, examples
// expected-no-diagnostics
[ms_example]
class C {};
// CHECK-NEXT: AnnotateAttr{{.*}} "ms_example"

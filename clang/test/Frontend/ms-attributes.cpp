// RUN: %clang -fplugin=%llvmshlibdir/MicrosoftAttributes%pluginext -fsyntax-only -fms-extensions -E %s | FileCheck %s
// REQUIRES: plugins, examples
// expected-no-diagnostics
[ms_example]
class C {};
// CHECK: AnnotateAttr{{.*}} "ms_example"

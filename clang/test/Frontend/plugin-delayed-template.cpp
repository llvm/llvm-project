// RUN: %clang_cc1 -fdelayed-template-parsing -load %llvmshlibdir/PrintFunctionNames%pluginext -plugin print-fns -plugin-arg-print-fns -parse-template -plugin-arg-print-fns ForcedTemplate %s 2>&1 | FileCheck %s
// REQUIRES: plugins, examples

template <typename T>
void TemplateDep();

// CHECK: top-level-decl: "ForcedTemplate"
// The plugin should force parsing of this template, even though it's
// not used and -fdelayed-template-parsing is specified.
// CHECK: warning: expression result unused
// CHECK: late-parsed-decl: "ForcedTemplate"
template <typename T>
void ForcedTemplate() {
  TemplateDep<T>();  // Shouldn't crash.

  "";  // Triggers the warning checked for above.
}

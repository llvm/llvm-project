// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
#define multiline_if_macro(c, t) \
if (c) { \
  return t; \
}

int testMacroLocations(void) {

  // Expanded macros will use the location of the expansion site.
  multiline_if_macro(1, 3);
  // CHECK: cir.scope {
  // CHECK:   cir.if %{{.+}} {
  // CHECK:     cir.return %{{.+}} : !s32i loc(#loc[[#LOC:]])
  // CHECK:   } loc(#loc[[#LOC]])
  // CHECK: } loc(#loc[[#LOC]])

  // Regular if statements should use different locations.
  if (1) {
    return 3;
  }
  //     CHECK: cir.scope {
  //     CHECK:   cir.if %{{.+}} {
  //     CHECK:     cir.return %{{.+}} : !s32i loc(#loc[[#LOC:]])
  // CHECK-NOT:   } loc(#loc[[#LOC]])
  // CHECK-NOT: } loc(#loc[[#LOC]])

  return 0;
}

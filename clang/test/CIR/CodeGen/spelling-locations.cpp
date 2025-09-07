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

void testIfStmtLocations(int f) {
  if (f)
    ;
  else
    ;

  if (f)
    ++f;
  else
    ;

  if (f)
    ;
  else
    --f;

  if (f)
    ++f;
  else
    --f;
}

// CHECK: cir.if %{{.+}} {
// CHECK: } else {
// CHECK: } loc(#loc[[#LOC1:]])

// CHECK: cir.if %{{.+}} {
// CHECK:   %{{.+}} = cir.load
// CHECK:   %{{.+}} = cir.unary(inc
// CHECK:   cir.store
// CHECK: } else {
// CHECK: } loc(#loc[[#LOC2:]])

// CHECK: cir.if %{{.+}} {
// CHECK: } else {
// CHECK:   %{{.+}} = cir.load
// CHECK:   %{{.+}} = cir.unary(dec
// CHECK:   cir.store
// CHECK: } loc(#loc[[#LOC3:]])

// CHECK: cir.if %{{.+}} {
// CHECK:   %{{.+}} = cir.load
// CHECK:   %{{.+}} = cir.unary(inc
// CHECK:   cir.store
// CHECK: } else {
// CHECK:   %{{.+}} = cir.load
// CHECK:   %{{.+}} = cir.unary(dec
// CHECK:   cir.store
// CHECK: } loc(#loc[[#LOC4:]])

// CHECK: #loc[[#LOC12:]] = loc({{.+}}:35:5)
// CHECK: #loc[[#LOC11:]] = loc({{.+}}:33:5)

// CHECK: #loc[[#LOC23:]] = loc({{.+}}:40:5)
// CHECK: #loc[[#LOC21:]] = loc({{.+}}:38:5)
// CHECK: #loc[[#LOC22:]] = loc({{.+}}:38:7)

// CHECK: #loc[[#LOC33:]] = loc({{.+}}:45:7)
// CHECK: #loc[[#LOC31:]] = loc({{.+}}:43:5)
// CHECK: #loc[[#LOC32:]] = loc({{.+}}:45:5)

// CHECK: #loc[[#LOC44:]] = loc({{.+}}:50:7)
// CHECK: #loc[[#LOC41:]] = loc({{.+}}:48:5)
// CHECK: #loc[[#LOC42:]] = loc({{.+}}:48:7)
// CHECK: #loc[[#LOC43:]] = loc({{.+}}:50:5)

// CHECK: #loc[[#LOC1]] = loc(fused[#loc[[#LOC11]], #loc[[#LOC12]]])
// CHECK: #loc[[#LOC2]] = loc(fused[#loc[[#LOC21]], #loc[[#LOC22]], #loc[[#LOC23]]])
// CHECK: #loc[[#LOC3]] = loc(fused[#loc[[#LOC31]], #loc[[#LOC32]], #loc[[#LOC33]]])
// CHECK: #loc[[#LOC4]] = loc(fused[#loc[[#LOC41]], #loc[[#LOC42]], #loc[[#LOC43]], #loc[[#LOC44]]])

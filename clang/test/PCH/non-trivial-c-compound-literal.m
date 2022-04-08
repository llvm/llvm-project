// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -x objective-c -fobjc-arc -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -x objective-c -fobjc-arc -include-pch %t -emit-llvm -o - %s | FileCheck %s

#ifndef HEADER
#define HEADER

typedef struct {
  id f;
} S;

static inline id getObj(id a) {
  S *p = &(S){ .f = a };
  return p->f;
}

#else

// CHECK: %[[STRUCT_S:.*]] = type { i8* }

// CHECK: define internal i8* @getObj(
// CHECK: %[[_COMPOUNDLITERAL:.*]] = alloca %[[STRUCT_S]],
// CHECK: %[[V5:.*]] = bitcast %[[STRUCT_S]]* %[[_COMPOUNDLITERAL]] to i8**
// CHECK: call void @__destructor_8_s0(i8** %[[V5]])

id test(id a) {
  return getObj(a);
}

#endif

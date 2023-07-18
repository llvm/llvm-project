// RUN: rm -rf %t
// RUN: %clang -fmodules-cache-path=%t -fmodules -x objective-c -I %S/Inputs -emit-ast -o %t.ast %s
// RUN: %clang_cc1 -ast-print -x ast - < %t.ast | FileCheck %s

@import import_decl;
// CHECK: struct T

int main(void) {
  return 0;
}

// <rdar://problem/15084587>
@interface A
-method;
@end

void testImport(A *import) {
  [import method];
}

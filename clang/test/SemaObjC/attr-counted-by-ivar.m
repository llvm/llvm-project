// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify=expected,eager %s
// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify -fexperimental-late-parse-attributes %s
// RUN: not %clang_cc1 -fsyntax-only -Wno-objc-root-class -ast-dump %s 2>/dev/null | FileCheck %s

// counted_by-family attributes apply to Objective-C instance variables, with
// a count that's an instance variable of the same ivar list.

#define __counted_by(f) __attribute__((counted_by(f)))
#define __sized_by(f) __attribute__((sized_by(f)))

int global; // #global

@interface I {
  int n; // #n
  // CHECK: ObjCIvarDecl {{.*}} p 'int * __counted_by(n)':'int *'
  int *__counted_by(n) p;
  // CHECK: ObjCIvarDecl {{.*}} q 'char * __sized_by(n)':'char *'
  char *__sized_by(n) q;
  // expected-error@+1{{'counted_by' argument must be a simple declaration reference}}
  int *__counted_by(n + 1) r;
  // expected-error@+2{{field 'global' in 'counted_by' not inside structure}}
  // expected-note@#global{{'global' declared here}}
  int *__counted_by(global) s;
  // FIXME: The late-parsed one is applied to the outer pointer instead.
  // eager-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int *__counted_by(n) *pp;
  // A struct declared in the ivar list can't count with an instance variable.
  struct T {
    // expected-error@+2{{'counted_by' field 'n' isn't within the same struct as the annotated pointer}}
    // expected-note@#n{{'n' declared here}}
    int *__counted_by(n) t;
  } st;
}
@end

@interface J
@end

@implementation J {
  int m;
  // CHECK: ObjCIvarDecl {{.*}} u 'int * __counted_by(m)':'int *'
  int *__counted_by(m) u;
}
@end

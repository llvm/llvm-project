// XFAIL: target={{.*}}-aix{{.*}}
// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

@interface MyCls
@end

@protocol P1,P2;

// CHECK: [[@LINE+1]]:6 | function/C | foo | c:@F@foo#*$objc(cs)MyCls# | __Z3fooP5MyCls | Decl | rel: 0
void foo(MyCls *o);
// CHECK: [[@LINE+1]]:6 | function/C | foo | c:@F@foo#*Qoobjc(pl)P1objc(pl)P2# | __Z3fooPU15objcproto2P12P211objc_object | Decl | rel: 0
void foo(id<P2, P1> o);

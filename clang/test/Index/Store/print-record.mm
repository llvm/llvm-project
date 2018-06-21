// RUN: rm -rf %t.idx
// RUN: %clang_cc1 %s -index-store-path %t.idx
// RUN: c-index-test core -print-record %t.idx | FileCheck %s

// XFAIL: linux

@class MyCls;

@interface MyCls
@end

// CHECK: [[@LINE+2]]:6 | function/C | c:@F@foo#*$objc(cs)MyCls# | Decl | rel: 0
// CHECK: [[@LINE+1]]:10 | class/ObjC | c:objc(cs)MyCls | Ref,RelCont | rel: 1
void foo(MyCls *p);


// RANGE-NOT: before_range
void before_range();

// RANGE: [[@LINE+1]]:6 | function/C | c:@F@in_range1# | Decl
void in_range1();
// RANGE: [[@LINE+1]]:6 | function/C | c:@F@in_range2# | Decl
void in_range2();

// RANGE-NOT: after_range
void after_range();

// RUN: c-index-test core -print-record %t.idx -filepath %s:21:23 | FileCheck -check-prefix=RANGE %s

// RUN: not %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c++ -fobjc-arc %s 2>&1 | FileCheck %s
// rdar://12788838

id obj;

void Test1() {
  void *foo = reinterpret_cast<void *>(obj);
}
// CHECK: {7:15-7:39}:"(__bridge void *)"
// CHECK: {7:15-7:39}:"(__bridge_retained void *)"

typedef const void * CFTypeRef;
extern "C" CFTypeRef CFBridgingRetain(id X);

void Test2() {
  void *foo = reinterpret_cast<void *>(obj);
}
// CHECK: {16:15-16:39}:"(__bridge void *)"
// CHECK: {16:15-16:39}:"CFBridgingRetain"

// RUN: not %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c++ -fobjc-arc %s 2>&1 | FileCheck %s

id obj;

void Test1() {
  void *foo = reinterpret_cast<void *>(obj);
}
// CHECK: {6:15-6:39}:"(__bridge void *)"
// CHECK: {6:15-6:39}:"(__bridge_retained void *)"

typedef const void * CFTypeRef;
extern "C" CFTypeRef CFBridgingRetain(id X);

void Test2() {
  void *foo = reinterpret_cast<void *>(obj);
}
// CHECK: {15:15-15:39}:"(__bridge void *)"
// CHECK: {15:15-15:39}:"CFBridgingRetain"

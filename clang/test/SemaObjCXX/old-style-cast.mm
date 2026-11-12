// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -Wold-style-cast -verify %s
// expected-no-diagnostics

// We don't currently have a way to write ARC/C++ bridge casts in terms of C++
// casts, so ensure we don't emit an old-style-cast warning in these cases.

id test(void *opaqueInput) {
  id someObjCObject = (__bridge id)opaqueInput;
  void *someCFObject = (__bridge_retained void *)someObjCObject;
  return (__bridge_transfer id)someCFObject;
}

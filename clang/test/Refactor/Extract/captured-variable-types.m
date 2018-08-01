
@interface Interface

@end

@protocol Protocol

@end

int basicObjCTypes(Interface *ip, id p, id<Protocol> pp, Interface<Protocol> *ipp) {
  return basicObjCTypes(ip, p, pp, ipp);
}
// CHECK1: "static int extracted(Interface *ip, Interface<Protocol> *ipp, id p, id<Protocol> pp) {\nreturn basicObjCTypes(ip, p, pp, ipp);\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK1-NEXT: "extracted(ip, ipp, p, pp)" [[@LINE-3]]:10 -> [[@LINE-3]]:40

// RUN: clang-refactor-test perform -action extract -selected=%s:11:10-11:40 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:11:10-11:40 %s -x objective-c++ | FileCheck --check-prefix=CHECK1 %s

typedef signed char BOOL;
#define YES __objc_yes
#define NO __objc_no

int boolType(BOOL b) {
  BOOL b2 = YES;
  return boolType(b && b2 && YES && NO);
}
// CHECK2: "static int extracted(BOOL b, BOOL b2) {\nreturn boolType(b && b2 && YES && NO);\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK2-NEXT: "extracted(b, b2)" [[@LINE-3]]:10 -> [[@LINE-3]]:40

// RUN: clang-refactor-test perform -action extract -selected=%s:25:10-25:40 %s | FileCheck --check-prefix=CHECK2 %s
;
int mutationOfObjCPointer(Interface *ip) {
  return ip = 0, mutationOfObjCPointer(ip);
}
// CHECK3-C: "static int extracted(Interface **ip) {\nreturn *ip = 0, mutationOfObjCPointer(*ip);\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK3-C-NEXT: "extracted(&ip)" [[@LINE-3]]:10 -> [[@LINE-3]]:43
// CHECK3-CPP: "static int extracted(Interface *&ip) {\nreturn ip = 0, mutationOfObjCPointer(ip);\n}\n\n" [[@LINE-5]]:1 -> [[@LINE-5]]:1
// CHECK3-CPP-NEXT: "extracted(ip)" [[@LINE-5]]:10 -> [[@LINE-5]]:43

// RUN: clang-refactor-test perform -action extract -selected=%s:33:10-33:43 %s | FileCheck --check-prefix=CHECK3-C %s
// RUN: clang-refactor-test perform -action extract -selected=%s:33:10-33:43 %s -x objective-c++ | FileCheck --check-prefix=CHECK3-CPP %s

void silenceStrongInARC() {
  Interface *pointer;
// silence-strong-in-arc-begin: +1:1
  mutationOfObjCPointer(pointer);
// silence-strong-in-arc-end: +0:1
// silence-strong2-in-arc-begin: +1:1
  pointer = 0;
// silence-strong2-in-arc-end: +0:1
  (void)pointer;
}
// CHECK-ARC: "static int extracted(Interface *pointer) {\nreturn mutationOfObjCPointer(pointer);\n}\n\n"
// CHECK-ARC: "static void extracted(Interface **pointer) {\n*pointer = 0;\n}\n\n"

// RUN: clang-refactor-test perform -action extract -selected=silence-strong-in-arc -selected=silence-strong2-in-arc %s -fobjc-arc | FileCheck --check-prefix=CHECK-ARC %s

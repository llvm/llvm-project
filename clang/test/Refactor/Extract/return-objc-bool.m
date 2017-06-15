#ifdef SIGNED
typedef signed char BOOL;
#else

#ifndef __cplusplus
#define bool _Bool
#endif

typedef bool BOOL;

#endif

#define YES __objc_yes
#define NO __objc_no

typedef struct {
  BOOL b;
} HasBool;

// Always prefer to use BOOL in the Objective-C methods.

@interface I
@end

@implementation I

- (BOOL)boolType:(BOOL)b with:(HasBool*)s {
  BOOL x = b && YES;
  BOOL y = [self boolType: b with: s];
  BOOL z = s->b;
  BOOL a = !b;
  return (b == NO);
}
// RUN: clang-refactor-test perform -action extract -selected=%s:28:12-28:20 -selected=%s:29:12-29:38 -selected=%s:30:12-30:16 -selected=%s:31:12-31:14 -selected=%s:32:10-32:19 %s | FileCheck --check-prefix=CHECKBOOL %s
// RUN: clang-refactor-test perform -action extract -selected=%s:28:12-28:20 -selected=%s:29:12-29:38 -selected=%s:30:12-30:16 -selected=%s:31:12-31:14 -selected=%s:32:10-32:19 %s -D SIGNED | FileCheck --check-prefix=CHECKBOOL %s
// RUN: clang-refactor-test perform -action extract -selected=%s:28:12-28:20 -selected=%s:29:12-29:38 -selected=%s:30:12-30:16 -selected=%s:31:12-31:14 -selected=%s:32:10-32:19 %s -x objective-c++ | FileCheck --check-prefix=CHECKBOOL %s
// RUN: clang-refactor-test perform -action extract -selected=%s:28:12-28:20 -selected=%s:29:12-29:38 -selected=%s:30:12-30:16 -selected=%s:31:12-31:14 -selected=%s:32:10-32:19 %s -x objective-c++ -D SIGNED | FileCheck --check-prefix=CHECKBOOL %s

// CHECKBOOL: "static BOOL extracted

#ifdef __cplusplus

// Prefer BOOL even in Objective-C++ methods.

- (BOOL)chooseBOOLEvenInCPlusPlus:(bool)b and:(bool)c {
  bool x = b && c;
  bool n = !b;
}

#endif
// RUN: clang-refactor-test perform -action extract -selected=%s:46:12-46:18 -selected=%s:47:12-47:14 %s -x objective-c++ | FileCheck --check-prefix=CHECKBOOL %s
// RUN: clang-refactor-test perform -action extract -selected=%s:46:12-46:18 -selected=%s:47:12-47:14 %s -x objective-c++ -D SIGNED | FileCheck --check-prefix=CHECKBOOL %s

@end

#ifdef __cplusplus

// In Objective-C++ functions/methods we want to pick the type based on the expression.

BOOL boolObjCFunction(BOOL b, BOOL c) {
  BOOL x = b && c;
  BOOL y = boolObjCFunction(b, c);
  return b;
}
// RUN: clang-refactor-test perform -action extract -selected=%s:61:12-61:18 -selected=%s:62:12-62:34 %s -x objective-c++ | FileCheck --check-prefix=CHECKBOOL %s
// RUN: clang-refactor-test perform -action extract -selected=%s:61:12-61:18 -selected=%s:62:12-62:34 %s -x objective-c++ -D SIGNED | FileCheck --check-prefix=CHECKBOOL %s

bool boolCPlusPlusFunction(bool b, bool c) {
  bool x = b && c;
  bool y = boolCPlusPlusFunction(b, c);
  return b;
}
// CHECKNORMAL: "static bool extracted
// RUN: clang-refactor-test perform -action extract -selected=%s:69:12-69:18 -selected=%s:70:12-70:39 %s -x objective-c++ | FileCheck --check-prefix=CHECKNORMAL %s

class AClass {
  AClass(BOOL b, BOOL c, bool d, bool e) {
    BOOL x = b && c;
    bool y = d && e;
  }
  void method(BOOL b, BOOL c, bool d, bool e) {
    BOOL x = b || c;
    bool y = d || e;
  }
};
// RUN: clang-refactor-test perform -action extract -selected=%s:78:14-78:20 -selected=%s:82:14-84:20 %s -x objective-c++ | FileCheck --check-prefix=CHECKBOOL %s
// RUN: clang-refactor-test perform -action extract -selected=%s:79:14-79:20 -selected=%s:83:14-83:20 %s -x objective-c++ | FileCheck --check-prefix=CHECKNORMAL %s

#endif

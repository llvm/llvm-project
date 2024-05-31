// RUN: %clang_analyze_cc1 -fblocks -verify %s 2>&1 \
// RUN:   -analyzer-display-progress \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-output=text \
// RUN: | FileCheck %s

#include "Inputs/system-header-simulator-objc.h"

void clang_analyzer_warnIfReached();

// expected-note@+2 {{[debug] analyzing from f}}
// expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
static void f(void) { clang_analyzer_warnIfReached(); }

@interface I: NSObject
-(void)instanceMethod:(int)arg1 with:(int)arg2;
+(void)classMethod;
@end

@implementation I
// expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
-(void)instanceMethod:(int)arg1 with:(int)arg2 { clang_analyzer_warnIfReached(); }

// expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
+(void)classMethod { clang_analyzer_warnIfReached(); }
@end

// expected-note@+1 3 {{[debug] analyzing from g}}
void g(I *i, int x, int y) {
  [I classMethod]; // expected-note {{Calling 'classMethod'}}
  [i instanceMethod: x with: y]; // expected-note {{Calling 'instanceMethod:with:'}}

  void (^block)(void);
  // expected-warning@+1 {{REACHABLE}} expected-note@+1 {{REACHABLE}}
  block = ^{ clang_analyzer_warnIfReached(); };
  block(); // expected-note {{Calling anonymous block}}
}

// CHECK: analyzer-display-progress.m f
// CHECK: analyzer-display-progress.m -[I instanceMethod:with:]
// CHECK: analyzer-display-progress.m +[I classMethod]
// CHECK: analyzer-display-progress.m g
// CHECK: analyzer-display-progress.m block (line: 35, col: 11)

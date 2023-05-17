// RUN: %clang_cc1 -fsyntax-only -verify %s -DSILENCE
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wlogical-op-parentheses
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wparentheses
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s -Wlogical-op-parentheses 2>&1 | FileCheck %s

#ifdef SILENCE
// expected-no-diagnostics
#endif

void logical_op_parentheses(unsigned i) {
	const unsigned t = 1;
  (void)(i ||
             i && i);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:20-[[@LINE-6]]:20}:")"

  (void)(t ||
             t && t);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:20-[[@LINE-6]]:20}:")"

  (void)(t && 
             t || t);
#ifndef SILENCE
  // expected-warning@-3 {{'&&' within '||'}}
  // expected-note@-4 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:10-[[@LINE-6]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:15-[[@LINE-6]]:15}:")"
	
	(void)(i || i && "w00t");
  (void)("w00t" && i || i);
  (void)("w00t" && t || t);
  (void)(t && t || 0);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:16-[[@LINE-6]]:16}:")"
  (void)(1 && t || t);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:16-[[@LINE-6]]:16}:")"
  (void)(0 || t && t);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:21-[[@LINE-6]]:21}:")"
  (void)(t || t && 1);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:21-[[@LINE-6]]:21}:")"

  (void)(i || i && "w00t" || i);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:26-[[@LINE-6]]:26}:")"

  (void)(i || "w00t" && i || i);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:26-[[@LINE-6]]:26}:")"

  (void)(i && i || 0);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:16-[[@LINE-6]]:16}:")"
  (void)(0 || i && i);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:21-[[@LINE-6]]:21}:")"
}

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -DCHECK_ERROR %s -verify 

float function_scope(float a) {
#pragma float_control(precise, on) junk // expected-warning {{extra tokens at end of '#pragma float_control' - ignored}}
  return a;
}

// Ok, at namespace scope.
namespace foo {
#pragma float_control(push)
#pragma float_control(pop)
}

// Ok, within a language linkage specification.
extern "C" {
#pragma float_control(push)
#pragma float_control(pop)
}

// Same.
extern "C++" {
#pragma float_control(push)
#pragma float_control(pop)
}

#ifdef CHECK_ERROR
// Ok at file scope.
#pragma float_control(push)
#pragma float_control(pop)
#pragma float_control(precise, on, push)
void check_stack() {
  // Not okay within a function declaration.
#pragma float_control(push)                   // expected-error {{can only appear at file or namespace scope or within a language linkage specification}}
#pragma float_control(pop)                    // expected-error {{can only appear at file or namespace scope or within a language linkage specification}}
#pragma float_control(precise, on, push)      // expected-error {{can only appear at file or namespace scope or within a language linkage specification}}
#pragma float_control(except, on, push)       // expected-error {{can only appear at file or namespace scope or within a language linkage specification}}
#pragma float_control(except, on, push, junk) // expected-error {{float_control is malformed}}
  return;
}
#endif

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fdenormal-fp-math=preserve-sign,preserve-sign -fsyntax-only %s -DDEFAULT -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only %s -ffp-contract=fast -DPRECISE -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only %s -ffp-contract=off -frounding-math -ffp-exception-behavior=strict -DSTRICT -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu -menable-no-infs -menable-no-nans -menable-unsafe-fp-math -fno-signed-zeros -mreassociate -freciprocal-math -ffp-contract=fast -ffast-math -ffinite-math-only -fsyntax-only %s -DFAST -verify
double a = 0.0;
double b = 1.0;

#ifdef STRICT
#pragma float_control(precise, off) // expected-error {{'#pragma float_control(precise, off)' is illegal when except is enabled}}
#else
#ifndef FAST
#pragma STDC FENV_ACCESS ON
#pragma float_control(precise, off) // expected-error {{'#pragma float_control(precise, off)' is illegal when except is enabled}}
#endif
#endif

#pragma float_control(precise, on)
#pragma float_control(except, on) // OK
#ifndef STRICT
#pragma float_control(except, on)
#pragma float_control(precise, off) // expected-error {{'#pragma float_control(precise, off)' is illegal when except is enabled}}
#endif
int main() {
#ifdef STRICT
#pragma float_control(precise, off) // expected-error {{'#pragma float_control(precise, off)' is illegal when except is enabled}}
#else
#pragma float_control(precise, off) // expected-error {{'#pragma float_control(precise, off)' is illegal when except is enabled}}
#endif
#pragma float_control(except, on)
  //  error: '#pragma float_control(except, on)' is illegal when precise is disabled
  double x = b / a; // only used for fp flag setting
  if (a == a)       // only used for fp flag setting
    return 0;       //(int)x;
}

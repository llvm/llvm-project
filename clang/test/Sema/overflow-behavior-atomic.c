// RUN: %clang_cc1 %s -fexperimental-overflow-behavior-types -verify -fsyntax-only -std=c11

// Overflow behavior types do not compose with _Atomic. Atomic
// read-modify-write operations lower to a single atomicrmw and are never
// instrumented, so an '__ob_trap' type would silently lose the overflow
// checking it promises. Reject the combination in every spelling.

typedef int __ob_trap tint;
typedef int __ob_wrap wint;
typedef _Atomic int aint;

// _Atomic applied over an overflow behavior type.
_Atomic tint a1;    // expected-error {{_Atomic cannot be applied to overflow behavior type 'tint' (aka '__ob_trap int')}}
_Atomic wint a2;    // expected-error {{_Atomic cannot be applied to overflow behavior type 'wint' (aka '__ob_wrap int')}}
_Atomic(tint) a3;   // expected-error {{_Atomic cannot be applied to overflow behavior type 'tint' (aka '__ob_trap int')}}
_Atomic(wint) a4;   // expected-error {{_Atomic cannot be applied to overflow behavior type 'wint' (aka '__ob_wrap int')}}

// Same, via the attribute spelling on the declarator.
_Atomic int __attribute__((overflow_behavior(trap))) a5;  // expected-error {{_Atomic cannot be applied to overflow behavior type '__ob_trap int'}}
_Atomic int __attribute__((overflow_behavior(wrap))) a6;  // expected-error {{_Atomic cannot be applied to overflow behavior type '__ob_wrap int'}}

// Pointers and members must be rejected too, not just plain declarations.
_Atomic tint *a7;             // expected-error {{_Atomic cannot be applied to overflow behavior type 'tint' (aka '__ob_trap int')}}
struct S { _Atomic tint m; }; // expected-error {{_Atomic cannot be applied to overflow behavior type 'tint' (aka '__ob_trap int')}}

// An overflow behavior specifier applied over an atomic type.
_Atomic __ob_trap int b1;   // expected-error {{__ob_trap specifier cannot be applied to atomic type '_Atomic(int)'}}
__ob_trap _Atomic int b2;   // expected-error {{__ob_trap specifier cannot be applied to atomic type '_Atomic(int)'}}
_Atomic __ob_wrap int b3;   // expected-error {{__ob_wrap specifier cannot be applied to atomic type '_Atomic(int)'}}
aint __ob_trap b4;          // expected-error {{__ob_trap specifier cannot be applied to atomic type 'aint'}}

// Same, via the attribute spelling.
aint __attribute__((overflow_behavior(trap))) b5; // expected-error {{'overflow_behavior' attribute cannot be applied to atomic type 'aint'}}
aint __attribute__((overflow_behavior(wrap))) b6; // expected-error {{'overflow_behavior' attribute cannot be applied to atomic type 'aint'}}

// Neither feature is disturbed on its own.
_Atomic int ok_atomic;
tint ok_trap;
wint ok_wrap;
_Atomic int *ok_atomic_ptr;
struct T { _Atomic int m; tint n; };

void uses(void) {
  ok_atomic++;
  ok_trap++;
  ok_wrap++;
}

// RUN: %clang_cc1 -verify=ref,both      %s                                         -fms-compatibility -fms-extensions -fcxx-exceptions
// RUN: %clang_cc1 -verify=expected,both %s -fexperimental-new-constant-interpreter -fms-compatibility -fms-extensions -fcxx-exceptions

/// Used to assert because the two parameters to _rotl do not have the same type.
static_assert(_rotl(0x01, 5) == 32);

static_assert(alignof(__unaligned int) == 1, "");

static_assert(__noop() == 0, "");

constexpr int noopIsActuallyNoop() {
    int a = 0;
    __noop(throw);
    __noop(++a);
    __noop(a = 100);
    return a;
}
static_assert(noopIsActuallyNoop() == 0);


auto foo() {
  const __INTPTR_TYPE__ bar = (__INTPTR_TYPE__)&&label; // both-note {{via initialization of variable 'bar' here}}
  return (void *)bar; // both-warning {{returning address of label, which is local}}
label:
} // both-warning {{C++23 extension}}



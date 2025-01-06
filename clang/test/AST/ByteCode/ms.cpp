// RUN: %clang_cc1 -verify=ref,both %s -fms-extensions -fcxx-exceptions
// RUN: %clang_cc1 -verify=expected,both %s -fexperimental-new-constant-interpreter -fms-extensions -fcxx-exceptions

// ref-no-diagnostics
// expected-no-diagnostics

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

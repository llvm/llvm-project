// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -triple x86_64-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -triple x86_64-linux-gnu %s -fexperimental-new-constant-interpreter

int n;
constexpr int *p = 0;
// expected-error@+1 {{must be initialized by a constant expression}}
constexpr int *k = (int *) __builtin_assume_aligned(p, 16, n = 5);

constexpr void *l = __builtin_assume_aligned(p, 16);

// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{cast from 'void *' is not allowed in a constant expression}}
constexpr int *c = (int *) __builtin_assume_aligned(p, 16);

// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{alignment of the base pointee object (4 bytes) is less than the asserted 16 bytes}}
constexpr void *m = __builtin_assume_aligned(&n, 16);

// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{offset of the aligned pointer from the base pointee object (-2 bytes) is not a multiple of the asserted 4 bytes}}
constexpr void *q1 = __builtin_assume_aligned(&n, 4, 2);
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{offset of the aligned pointer from the base pointee object (2 bytes) is not a multiple of the asserted 4 bytes}}
constexpr void *q2 = __builtin_assume_aligned(&n, 4, -2);
constexpr void *q3 = __builtin_assume_aligned(&n, 4, 4);
constexpr void *q4 = __builtin_assume_aligned(&n, 4, -4);

static char ar1[6];
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{alignment of the base pointee object (1 byte) is less than the asserted 16 bytes}}
constexpr void *r1 = __builtin_assume_aligned(&ar1[2], 16);

static char ar2[6] __attribute__((aligned(32)));
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{offset of the aligned pointer from the base pointee object (2 bytes) is not a multiple of the asserted 16 bytes}}
constexpr void *r2 = __builtin_assume_aligned(&ar2[2], 16);
constexpr void *r3 = __builtin_assume_aligned(&ar2[2], 16, 2);
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{offset of the aligned pointer from the base pointee object (1 byte) is not a multiple of the asserted 16 bytes}}
constexpr void *r4 = __builtin_assume_aligned(&ar2[2], 16, 1);

constexpr int* x = __builtin_constant_p((int*)0xFF) ? (int*)0xFF : (int*)0xFF;
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{value of the aligned pointer (255) is not a multiple of the asserted 32 bytes}}
constexpr void *s1 = __builtin_assume_aligned(x, 32);
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{value of the aligned pointer (250) is not a multiple of the asserted 32 bytes}}
constexpr void *s2 = __builtin_assume_aligned(x, 32, 5);
constexpr void *s3 = __builtin_assume_aligned(x, 32, -1);


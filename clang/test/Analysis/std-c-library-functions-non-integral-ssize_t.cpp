// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -triple x86_64-unknown-linux -verify

// expected-no-diagnostics

// Regression test for https://github.com/llvm/llvm-project/issues/209436
// When ssize_t is defined as a non-integral type (violating POSIX), the
// checker should gracefully skip the summary rather than asserting.

struct sockaddr;
using socklen_t = unsigned;
typedef decltype(sizeof(int)) size_t;
typedef float ssize_t; // Non-POSIX-conforming definition.
ssize_t recvfrom(int socket, void *buffer, size_t length, int flags,
                 struct sockaddr *address, socklen_t *address_len);

int bar(void);
void foo() { bar(); }

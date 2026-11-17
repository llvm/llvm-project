// RUN: %clang_cc1 -fsyntax-only -triple hexagon -target-cpu hexagonv85 -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -verify %s
// expected-no-diagnostics

// Without HLX, vectors of HLX sizes are ordinary vectors.

typedef short V2K __attribute__((__vector_size__(2048)));
typedef long V4K __attribute__((__vector_size__(4096)));

V2K g;
struct S { V2K v; };

long f(V2K v, V4K *p) {
  V4K vp = *p;
  return v[2] + vp[1];
}

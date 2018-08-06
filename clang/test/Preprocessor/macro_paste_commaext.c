// RUN: %clang_cc1 %s -E | FileCheck --strict-whitespace --match-full-lines %s

// In the following tests, note that the output is sensitive to the
// whitespace *preceding* the varargs argument, as well as to
// interior whitespace. AFAIK, this is the only case where whitespace
// preceding an argument matters, and might be considered a bug in
// GCC. Nevertheless, since this feature is a GCC extension in the
// first place, we'll follow along.

#define debug(format, ...) format, ## __VA_ARGS__)
// CHECK:V);
debug(V);
// CHECK:W,1, 2);
debug(W,1, 2);
// CHECK:X, 1, 2);
debug(X, 1, 2 );
// CHECK:Y,);
debug(Y, );
// CHECK:Z,);
debug(Z,);

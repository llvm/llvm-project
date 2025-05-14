// RUN: %clang --analyze %s -ftime-trace -Xclang -verify
// expected-no-diagnostics

// GitHub issue 139779
struct {} a; // no-crash

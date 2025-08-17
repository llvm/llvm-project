// RUN: %clang_analyze_cc1 -analyzer-checker=core,apiModeling %s -ftime-trace=%t.raw.json -verify
// expected-no-diagnostics

// GitHub issue 139779
struct {} a; // no-crash

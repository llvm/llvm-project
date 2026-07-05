// Test with PCH
// RUN: %clang_cc1 -std=c++11 -x c++-header -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s
// expected-no-diagnostics

// PR10847
#ifndef HEADER
#define HEADER
struct NSSize {
  double width;
  double height;
};
typedef struct NSSize NSSize;

static inline NSSize NSMakeSize(double w, double h) {
    NSSize s = { w, h };
    return s;
}
#else
float test(float v1, float v2) {
	NSSize s = NSMakeSize(v1, v2);
	return s.width;
}
#endif

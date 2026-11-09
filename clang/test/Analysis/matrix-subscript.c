// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -fenable-matrix %s

// expected-no-diagnostics

typedef float m2 __attribute__((matrix_type(2, 2)));

float load(m2 m) {
	return m[0][0]; // no-crash
}

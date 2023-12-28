// RUN: %clang_cc1 -fsyntax-only -Warray-bounds -verify %s

typedef double float64x1_t __attribute__ ((__vector_size__ (sizeof (double))));
void foo(double i)
{
    float64x1_t j = {i}; // expected-note 2 {{vector 'j' declared here}}
    double U = j[0];
    double V = j[1]; // expected-warning {{vector index 1 is past the end of the vector (that has type '__attribute__((__vector_size__(1 * sizeof(double)))) double' (vector of 1 'double' value))}}
    double W = j[-1]; // expected-warning {{vector index -1 is before the beginning of the vector}}
}

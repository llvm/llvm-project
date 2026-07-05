// RUN: %clang_cc1 -Wall -fsyntax-only %s -std=c++11 -verify

// The rdar11671507_vector<int *>[]> would previously crash CFG construction
// because of the temporary array of vectors.
template <typename T>
class rdar11671507_vector {
public:
  rdar11671507_vector();
  ~rdar11671507_vector();
  T *Base;
  T *End;
};

void rdar11671507(rdar11671507_vector<int*> v, rdar11671507_vector<int*> w) {
  for (auto &vec : (rdar11671507_vector<int *>[]){ v, w }) {} // expected-warning {{unused}}
}

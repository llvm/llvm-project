// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

void __libcpp_verbose_abort(const char *__format, ...);

using size_t = __typeof(sizeof(int));
namespace std{
template <class T, size_t N>
class array {
  T elements[N];
  
  public:
  T& operator[](unsigned i) {
    if (i >= N) {
      __libcpp_verbose_abort("%s", "aborting");
    }
    return elements[i];
  }
};
}

class ArrayClass {
public:
  void ref() const;
  void deref() const;
  typedef std::array<std::array<double, 4>, 4> Matrix;
  double e() { return matrix[3][0]; }
  Matrix matrix;
};

class AnotherClass {
  RefPtr<ArrayClass> matrix;
  void test() {
    double val = { matrix->e()};
  }
};


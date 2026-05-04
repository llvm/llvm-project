// Test this without pch.
// RUN: %clang_cc1 -x c++ -std=c++11 -DHEADER -DSOURCE -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++11 -DHEADER -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++11 -DHEADER -include-pch %t -emit-llvm -o - %s 

// RUN: %clang_cc1 -x c++ -std=c++11 -DHEADER -emit-pch -fpch-instantiate-templates -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++11 -DHEADER -include-pch %t -emit-llvm -o - %s

#ifdef HEADER
int n;
struct S {
  int *p = &m;
  int &m = n;
  S *that = this;
};
template<typename T> struct X { T t {0}; };

struct v_t { };

struct m_t
{
    struct { v_t v; };
    m_t() { }
};

#endif

#ifdef SOURCE
S s;

struct E { explicit E(int); };
X<E> x;

m_t *test() {
  return new m_t;
}

#elif HEADER
#undef HEADER
#define SOURCE
#endif

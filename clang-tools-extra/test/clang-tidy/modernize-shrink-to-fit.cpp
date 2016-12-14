// RUN: %check_clang_tidy %s modernize-shrink-to-fit %t

namespace std {
template <typename T> struct vector { void swap(vector &other); };
}

void f() {
  std::vector<int> v;

  std::vector<int>(v).swap(v);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the shrink_to_fit method should be used to reduce the capacity of a shrinkable container [modernize-shrink-to-fit]
  // CHECK-FIXES: {{^  }}v.shrink_to_fit();{{$}}

  std::vector<int> &vref = v;
  std::vector<int>(vref).swap(vref);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the shrink_to_fit method should
  // CHECK-FIXES: {{^  }}vref.shrink_to_fit();{{$}}

  std::vector<int> *vptr = &v;
  std::vector<int>(*vptr).swap(*vptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the shrink_to_fit method should
  // CHECK-FIXES: {{^  }}vptr->shrink_to_fit();{{$}}
}

struct X {
  std::vector<int> v;
  void f() {
    std::vector<int>(v).swap(v);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: the shrink_to_fit method should
    // CHECK-FIXES: {{^    }}v.shrink_to_fit();{{$}}

    std::vector<int> *vptr = &v;
    std::vector<int>(*vptr).swap(*vptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: the shrink_to_fit method should
    // CHECK-FIXES: {{^    }}vptr->shrink_to_fit();{{$}}
  }
};

template <typename T> void g() {
  std::vector<int> v;
  std::vector<int>(v).swap(v);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the shrink_to_fit method should
  // CHECK-FIXES: {{^  }}v.shrink_to_fit();{{$}}

  std::vector<T> v2;
  std::vector<T>(v2).swap(v2);
  // CHECK-FIXES: {{^  }}std::vector<T>(v2).swap(v2);{{$}}
}

template <typename T> void g2() {
  std::vector<int> v;
  std::vector<int>(v).swap(v);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the shrink_to_fit method should
  // CHECK-FIXES: {{^  }}v.shrink_to_fit();{{$}}

  T v3;
  T(v3).swap(v3);
  // CHECK-FIXES: {{^  }}T(v3).swap(v3);{{$}}
}

#define COPY_AND_SWAP_INT_VEC(x) std::vector<int>(x).swap(x)
// CHECK-FIXES: #define COPY_AND_SWAP_INT_VEC(x) std::vector<int>(x).swap(x)

void h() {
  g<int>();
  g<double>();
  g<bool>();
  g2<std::vector<int>>();
  std::vector<int> v;
  COPY_AND_SWAP_INT_VEC(v);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the shrink_to_fit method should
  // CHECK-FIXES: {{^  }}COPY_AND_SWAP_INT_VEC(v);{{$}}
}


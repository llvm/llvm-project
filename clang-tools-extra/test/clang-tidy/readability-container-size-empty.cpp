// RUN: %check_clang_tidy %s readability-container-size-empty %t

namespace std {
template <typename T> struct vector {
  vector() {}
  unsigned long size() const {}
  bool empty() const {}
};
}

int main() {
  std::vector<int> vect;
  if (vect.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used to check for emptiness instead of 'size' [readability-container-size-empty]
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (vect.size() != 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (0 == vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (0 != vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (vect.size() > 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (0 < vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (vect.size() < 1)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (1 > vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (vect.size() >= 1)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (1 <= vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}
  if (vect.size() > 1) // no warning
    ;
  if (1 < vect.size()) // no warning
    ;
  if (!vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect.empty()){{$}}
  if (vect.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect.empty()){{$}}

  if (vect.empty())
    ;

  const std::vector<int> vect2;
  if (vect2.size() != 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!vect2.empty()){{$}}

  std::vector<int> *vect3 = new std::vector<int>();
  if (vect3->size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect3->empty()){{$}}

  delete vect3;

  const std::vector<int> &vect4 = vect2;
  if (vect4.size() == 0)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (vect4.empty()){{$}}
}

#define CHECKSIZE(x) if (x.size())
// CHECK-FIXES: #define CHECKSIZE(x) if (x.size())

template <typename T> void f() {
  std::vector<T> v;
  if (v.size())
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: the 'empty' method should be used
  // CHECK-FIXES: {{^  }}if (!v.empty()){{$}}
  // CHECK-FIXES-NEXT: ;
  CHECKSIZE(v);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: the 'empty' method should be used
  // CHECK-MESSAGES: CHECKSIZE(v);
}

void g() {
  f<int>();
  f<double>();
  f<char *>();
}

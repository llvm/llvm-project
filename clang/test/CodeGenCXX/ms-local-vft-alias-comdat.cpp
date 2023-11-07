// RUN: %clang_cc1 -fcxx-exceptions -triple=x86_64-windows-msvc  \
// RUN:  -Wmicrosoft-template -fms-compatibility -emit-llvm %s -o - \
// RUN:  | FileCheck %s

template <typename a> struct T {
  virtual void c();
  T(a h) {}
};
struct m {
  template <typename j> void ab(j ac) {
    using ad = T<j>;
    ad j(ac);
  }
};
template <typename ae> struct n {
  template <typename j> n(j ac) { q.ab(ac); }
  ae q;
};
class s : n<m> {
  using ag = n<m>;
public:
  template <typename j> s(j ac) : ag(ac) {}
};
struct ah {
  ah(s);
} a([]{});

//CHECK: "??_7?$T@V
//CHECK-SAME: <lambda_[[UNIQ:.*]]_0>
//CHECK-NOT: <lambda_0>
//CHECK-SAME: @@@@6B@"

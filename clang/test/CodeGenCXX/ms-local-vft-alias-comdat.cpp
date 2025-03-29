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

//CHECK: @0 = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4?$T@V<lambda_0>@@@@6B@", ptr @"?c@?$T@V<lambda_0>@@@@UEAAXXZ"] }
//CHECK: @"??_7?$T@V<lambda_0>@@@@6B@" = internal unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @0, i32 0, i32 0, i32 1)
//CHECK-NOT : "??_7?$e@V<lambda_0>@@@@6B@" = comdat any

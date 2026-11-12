// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits -fsafe-buffer-usage-suggestions %s 2>&1 | FileCheck %s

namespace NS1 {
  void foo(int *);
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:3-[[@LINE-1]]:3}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
  // CHECK-DAG: fix-it:"{{.*}}:{[[@LINE-2]]:18-[[@LINE-2]]:18}:";\nvoid foo(std::span<int>)"
  namespace NS2 {
    void foo(int *);
    // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:5-[[@LINE-1]]:5}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
    // CHECK-DAG: fix-it:"{{.*}}:{[[@LINE-2]]:20-[[@LINE-2]]:20}:";\nvoid foo(std::span<int>)"
    namespace NS3 {
      void foo(int *);
      // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:7-[[@LINE-1]]:7}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
      // CHECK-DAG: fix-it:"{{.*}}:{[[@LINE-2]]:22-[[@LINE-2]]:22}:";\nvoid foo(std::span<int>)"
    }
  }

  typedef int MyType;
}

void NS1::foo(int *p) {
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:15-[[@LINE-1]]:21}:"std::span<int> p"
  int tmp;
  tmp = p[5];
}
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void NS1::foo(int *p) {return NS1::foo(std::span<int>(p, <# size #>));}\n"

void NS1::NS2::foo(int *p) {
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:20-[[@LINE-1]]:26}:"std::span<int> p"
  int tmp;
  tmp = p[5];
}
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void NS1::NS2::foo(int *p) {return NS1::NS2::foo(std::span<int>(p, <# size #>));}\n"

void NS1::NS2::NS3::foo(int *p) {
  // CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:25-[[@LINE-1]]:31}:"std::span<int> p"
  int tmp;
  tmp = p[5];
}
// CHECK-DAG: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void NS1::NS2::NS3::foo(int *p) {return NS1::NS2::NS3::foo(std::span<int>(p, <# size #>));}\n"


void f(NS1::MyType * x) {
  // CHECK: fix-it:{{.*}}:{[[@LINE-1]]:8-[[@LINE-1]]:23}:"std::span<NS1::MyType> x"
  NS1::MyType tmp;
  tmp = x[5];
}
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:2-[[@LINE-1]]:2}:"\n{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} void f(NS1::MyType * x) {return f(std::span<NS1::MyType>(x, <# size #>));}\n"

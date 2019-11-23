// RUN: %check_clang_tidy %s llvm-namespace-comment %t

namespace n1 {
namespace n2 {
  void f();


  // CHECK-MESSAGES: :[[@LINE+2]]:1: warning: namespace 'n2' not terminated with a closing comment [llvm-namespace-comment]
  // CHECK-MESSAGES: :[[@LINE+1]]:2: warning: namespace 'n1' not terminated with a closing comment [llvm-namespace-comment]
}}
// CHECK-FIXES: } // namespace n2
// CHECK-FIXES: } // namespace n1

#define MACRO macro_expansion
namespace MACRO {
  void f();
  // CHECK-MESSAGES: :[[@LINE+1]]:1: warning: namespace 'MACRO' not terminated with a closing comment [llvm-namespace-comment]
}
// CHECK-FIXES: } // namespace MACRO

namespace MACRO {
  void g();
} // namespace MACRO

namespace MACRO {
  void h();
  // CHECK-MESSAGES: :[[@LINE+1]]:2: warning: namespace 'MACRO' ends with a comment that refers to an expansion of macro [llvm-namespace-comment]
} // namespace macro_expansion
// CHECK-FIXES: } // namespace MACRO

namespace n1 {
namespace MACRO {
namespace n2 {
  void f();
  // CHECK-MESSAGES: :[[@LINE+3]]:1: warning: namespace 'n2' not terminated with a closing comment [llvm-namespace-comment]
  // CHECK-MESSAGES: :[[@LINE+2]]:2: warning: namespace 'MACRO' not terminated with a closing comment [llvm-namespace-comment]
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: namespace 'n1' not terminated with a closing comment [llvm-namespace-comment]
}}}
// CHECK-FIXES: } // namespace n2
// CHECK-FIXES: } // namespace MACRO
// CHECK-FIXES: } // namespace n1

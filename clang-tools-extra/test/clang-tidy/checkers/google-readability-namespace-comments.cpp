// RUN: %check_clang_tidy %s google-readability-namespace-comments %t

namespace n1 {
namespace n2 {


void f(); // So that the namespace isn't empty.


// CHECK-MESSAGES: :[[@LINE+4]]:1: warning: namespace 'n2' not terminated with a closing comment [google-readability-namespace-comments]
// CHECK-MESSAGES: :[[@LINE-7]]:11: note: namespace 'n2' starts here
// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: namespace 'n1' not terminated with
// CHECK-MESSAGES: :[[@LINE-10]]:11: note: namespace 'n1' starts here
}}
// CHECK-FIXES: }  // namespace n2
// CHECK-FIXES: }  // namespace n1

#define MACRO macro_expansion
namespace MACRO {
void f(); // So that the namespace isn't empty.
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// CHECK-MESSAGES: :[[@LINE+2]]:1: warning: namespace 'MACRO' not terminated with
// CHECK-MESSAGES: :[[@LINE-10]]:11: note: namespace 'MACRO' starts here
}
// CHECK-FIXES: }  // namespace MACRO

namespace short1 {
namespace short2 {
// Namespaces covering 10 lines or fewer are exempt from this rule.





}
}

namespace n3 {









}; // namespace n3

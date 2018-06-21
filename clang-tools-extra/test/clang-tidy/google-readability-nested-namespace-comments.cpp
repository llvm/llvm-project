// RUN: %check_clang_tidy %s google-readability-namespace-comments %t

namespace n1::n2 {
namespace n3 {

// So that namespace is not empty.
void f();


// CHECK-MESSAGES: :[[@LINE+4]]:2: warning: namespace 'n3' not terminated with
// CHECK-MESSAGES: :[[@LINE-7]]:11: note: namespace 'n3' starts here
// CHECK-MESSAGES: :[[@LINE+2]]:3: warning: namespace 'n1::n2' not terminated with a closing comment [google-readability-namespace-comments]
// CHECK-MESSAGES: :[[@LINE-10]]:11: note: namespace 'n1::n2' starts here
}}
// CHECK-FIXES: }  // namespace n3
// CHECK-FIXES: }  // namespace n1::n2


// RUN: %check_clang_tidy %s google-readability-namespace-comments %t

namespace n1::n2 {
namespace /*comment1*/n3/*comment2*/::/*comment3*/inline/*comment4*/n4/*comment5*/ {

// So that namespace is not empty.
void f();


// CHECK-MESSAGES: :[[@LINE+4]]:1: warning: namespace 'n3::inline n4' not terminated with
// CHECK-MESSAGES: :[[@LINE-7]]:23: note: namespace 'n3::inline n4' starts here
// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: namespace 'n1::n2' not terminated with a closing comment [google-readability-namespace-comments]
// CHECK-MESSAGES: :[[@LINE-10]]:11: note: namespace 'n1::n2' starts here
}}
// CHECK-FIXES: }  // namespace n3::inline n4
// CHECK-FIXES: }  // namespace n1::n2

namespace n7::inline n8 {
// make namespace above 10 lines










} // namespace n7::inline n8

namespace n9::inline n10 {
// make namespace above 10 lines










} // namespace n9::n10
// CHECK-MESSAGES: :[[@LINE-1]]:2: warning: namespace 'n9::inline n10' ends with a comment that refers to a wrong namespace 'n9::n10' [google-readability-namespace-comments]

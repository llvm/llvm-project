// RUN: %check_clang_tidy %s google-readability-namespace-comments %t

namespace n1::n2 {
namespace /*comment1*/n3/*comment2*/::/*comment3*/inline/*comment4*/n4/*comment5*/ {

// So that namespace is not empty.
void f();


// CHECK-MESSAGES: :[[@LINE+4]]:1: warning: namespace 'n3::n4' not terminated with
// CHECK-MESSAGES: :[[@LINE-7]]:23: note: namespace 'n3::n4' starts here
// CHECK-MESSAGES: :[[@LINE+2]]:2: warning: namespace 'n1::n2' not terminated with a closing comment [google-readability-namespace-comments]
// CHECK-MESSAGES: :[[@LINE-10]]:11: note: namespace 'n1::n2' starts here
}}
// CHECK-FIXES: }  // namespace n3::n4
// CHECK-FIXES: }  // namespace n1::n2


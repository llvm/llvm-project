// RUN: %check_clang_tidy %s google-readability-namespace-comments %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     google-readability-namespace-comments.AllowNoNamespaceComments: true, \
// RUN:   }}'

namespace n1::n2 {
namespace /*comment1*/n3/*comment2*/::/*comment3*/inline/*comment4*/n4/*comment5*/ {

// So that namespace is not empty.
void f();


}}

namespace n7::inline n8 {
// make namespace above 10 lines

void hh();

// 1
// 2
// 3
// 4
// 5
// 6
// 7
} // namespace n7::inline n8

namespace n9::inline n10 {
// make namespace above 10 lines
void hh();
// 1
// 2
// 3
// 4
// 5
// 6
// 7
} // namespace n9::n10
// CHECK-MESSAGES: :[[@LINE-1]]:2: warning: namespace 'n9::inline n10' ends with a comment that refers to a wrong namespace 'n9::n10' [google-readability-namespace-comments]


namespace n11::n12 {
// make namespace above 10 lines
void hh();
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// CHECK-MESSAGES: :[[@LINE+1]]:2: warning: namespace 'n11::n12' ends with a comment that refers to a wrong namespace 'n1::n2' [google-readability-namespace-comments]
}; // namespace n1::n2
// CHECK-FIXES: }  // namespace n11::n12

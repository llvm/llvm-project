// RUN: %check_clang_tidy %s google-readability-namespace-comments %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     google-readability-namespace-comments.AllowOmittingNamespaceComments: true, \
// RUN:     google-readability-namespace-comments.ShortNamespaceLines: 0, \
// RUN:   }}'

// accept if namespace comments are fully omitted
namespace n1 {
namespace /* a comment */ n2 /* another comment */ {
void f();
}}

#define MACRO macro_expansion
namespace MACRO {
void f();
}

namespace [[deprecated("foo")]] namespace_with_attr {
inline namespace inline_namespace {
void f();
}
}

namespace [[]] {
void f();
}

// accept if namespace comments are partly omitted (e.g. only for nested namespace)
namespace n3 {
namespace n4 {
void f();
} // n4
}

// fail if namespace comment is different than expected
namespace n1 {
void f();
} // namespace n2
// CHECK-MESSAGES: :[[@LINE-1]]:2: warning: namespace 'n1' ends with a comment that refers to a wrong namespace 'n2' [google-readability-namespace-comments]


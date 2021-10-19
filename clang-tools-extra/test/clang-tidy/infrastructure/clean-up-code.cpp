// RUN: %check_clang_tidy %s misc-unused-using-decls %t -- --fix-notes
// RUN: %check_clang_tidy %s misc-unused-using-decls %t -- --fix-notes -format-style=none --
// RUN: %check_clang_tidy %s misc-unused-using-decls %t -- --fix-notes -format-style=llvm --
namespace a { class A {}; }
namespace b {
using a::A;
}
namespace c {}
// CHECK-MESSAGES: :[[@LINE-3]]:10: warning: using decl 'A' is unused [misc-unused-using-decls]
// CHECK-FIXES: {{^namespace a { class A {}; }$}}
// CHECK-FIXES-NOT: namespace
// CHECK-FIXES: {{^namespace c {}$}}
// FIXME: cleanupAroundReplacements leaves whitespace. Otherwise we could just
// check the next line.

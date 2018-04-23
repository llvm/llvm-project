// RUN: %check_clang_tidy -check-suffix=USING-A %s misc-unused-using-decls %t -- -- -DUSING_A
// RUN: %check_clang_tidy -check-suffix=USING-B %s misc-unused-using-decls %t -- -- -DUSING_B
// RUN: %check_clang_tidy %s misc-unused-using-decls %t

namespace a {class A {}; class B {}; class C {}; }
namespace b {
#if defined(USING_A)
using a::A;
#elif  defined(USING_B)
using a::B;
#else
using a::C;
#endif
}
namespace c {}
// CHECK-MESSAGES-USING-A: :[[@LINE-8]]:10: warning: using decl 'A' {{.*}}
// CHECK-MESSAGES-USING-B: :[[@LINE-7]]:10: warning: using decl 'B' {{.*}}
// CHECK-MESSAGES: :[[@LINE-6]]:10: warning: using decl 'C' {{.*}}
// CHECK-FIXES-USING-A-NOT: using a::A;$
// CHECK-FIXES-USING-B-NOT: using a::B;$
// CHECK-FIXES-NOT: using a::C;$

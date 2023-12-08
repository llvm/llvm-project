// RUN: %check_clang_tidy -check-suffix=USING-A %s misc-unused-using-decls %t -- -- -DUSING_A
// RUN: %check_clang_tidy -check-suffix=USING-B %s misc-unused-using-decls %t -- -- -DUSING_B
// RUN: %check_clang_tidy -check-suffix=USING-C,USING-D %s misc-unused-using-decls %t -- -- -DUSING_C_D
// RUN: %check_clang_tidy -check-suffixes=USING-C,USING-D %s misc-unused-using-decls %t -- -- -DUSING_C_D
// RUN: %check_clang_tidy %s misc-unused-using-decls %t

namespace a {class A {}; class B {}; class C {}; class D {}; class E {};}
namespace b {
#if defined(USING_A)
using a::A;
#elif  defined(USING_B)
using a::B;
#elif  defined(USING_C_D)
using a::C;
using a::D;
#else
using a::E;
#endif
}
namespace c {}
// CHECK-MESSAGES-USING-A: warning: using decl 'A' {{.*}}
// CHECK-MESSAGES-USING-B: warning: using decl 'B' {{.*}}
// CHECK-MESSAGES-USING-C: warning: using decl 'C' {{.*}}
// CHECK-MESSAGES-USING-D: warning: using decl 'D' {{.*}}
// CHECK-MESSAGES: warning: using decl 'E' {{.*}}
// CHECK-FIXES-USING-A-NOT: using a::A;$
// CHECK-FIXES-USING-B-NOT: using a::B;$
// CHECK-FIXES-USING-C-NOT: using a::C;$
// CHECK-FIXES-USING-C-NOT: using a::D;$
// CHECK-FIXES-USING-D-NOT: using a::C;$
// CHECK-FIXES-USING-D-NOT: using a::D;$
// CHECK-FIXES-NOT: using a::E;$

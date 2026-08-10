// Test this without pch.
// RUN: %clang_cc1 -x c++ -include %S/Inputs/namespaces.h -fsyntax-only %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -emit-pch -o %t %S/Inputs/namespaces.h
// RUN: %clang_cc1 -x c++ -include-pch %t -fsyntax-only %s 

int int_val;
N1::t1 *ip1 = &int_val;
N1::t2 *ip2 = &int_val;
N2::Inner::t3 *ip3 = &int_val;

float float_val;
namespace N2 { }
N2::t1 *fp1 = &float_val;

Alias1::t3 *ip4 = &int_val;
t3 *ip5 = &int_val;

void(*funp1)() = anon;

namespace {
  class C;
}
C* cp1;

namespace N3 {
  namespace {
    class C;
  }
}

N3::C *cp2;

void(*funp2)() = ext;

using N1::used_func;
void (*pused)() = used_func;

using N1::used_cls;
used_cls s1;
used_cls* ps1 = &s1;

inline namespace N4 { }
struct MemberOfN4 *mn4;

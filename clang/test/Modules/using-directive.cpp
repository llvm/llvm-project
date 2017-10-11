// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fno-modules-error-recovery -fno-spell-checking -verify %s

#pragma clang module build a
module a { explicit module b {} explicit module c {} }
#pragma clang module contents

#pragma clang module begin a.b
namespace b { int n; }
#pragma clang module end

#pragma clang module begin a.c
#pragma clang module import a.b
namespace c { using namespace b; }
#pragma clang module end

#pragma clang module begin a
#pragma clang module import a.c
using namespace c;
#pragma clang module end

#pragma clang module endbuild

#pragma clang module import a.b
void use1() {
  (void)n; // expected-error {{use of undeclared identifier}}
  (void)::n; // expected-error {{no member named 'n' in the global namespace}}
  (void)b::n;
}
namespace b {
  void use1_in_b() { (void)n; }
}
namespace c {
  void use1_in_c() { (void)n; } // expected-error {{use of undeclared identifier}}
}

#pragma clang module import a.c
void use2() {
  (void)n; // expected-error {{use of undeclared identifier}}
  (void)::n; // expected-error {{no member named 'n' in the global namespace}}
  (void)b::n;
  (void)c::n;
}
namespace b {
  void use2_in_b() { (void)n; }
}
namespace c {
  void use2_in_c() { (void)n; }
}

#pragma clang module import a
void use3() {
  (void)n;
  (void)::n;
  (void)b::n;
  (void)c::n;
}
namespace b {
  void use3_in_b() { (void)n; }
}
namespace c {
  void use3_in_c() { (void)n; }
}

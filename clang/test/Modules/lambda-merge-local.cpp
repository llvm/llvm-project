// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -verify %s -o -

#pragma clang module build M
module M { module X {} module Y {} }
#pragma clang module contents
#pragma clang module begin M.X
inline auto a = [] {};
#pragma clang module end
#pragma clang module begin M.Y
inline auto a = [] {};
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import M.X
#pragma clang module import M.Y

// expected-no-diagnostics
void use_a() { a(); }

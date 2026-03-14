// RUN: %clang_cc1 -fmodules %s -verify
// expected-no-diagnostics

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
namespace N { class X; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
namespace N { class Friendly { friend class X; }; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build C
module C {}
#pragma clang module contents
#pragma clang module begin C
#pragma clang module import A
void use_X(N::X *p);
#pragma clang module import B
// UsingShadowDecl names the friend declaration
using N::X;
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import B
namespace N { class AlsoFriendly { friend class X; }; }
#pragma clang module import A
#pragma clang module import C
// The friend declaration from N::Friendly is now the first in the redecl
// chain, so is not ordinarily visible. We need the IDNS of the UsingShadowDecl
// to still consider it to be visible, though.
X *p;

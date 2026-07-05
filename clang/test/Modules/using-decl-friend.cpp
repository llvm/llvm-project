// RUN: %clang_cc1 -fmodules %s -verify
// expected-no-diagnostics

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
namespace N {
  class X;
}
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {
  module X {}
  module Y {}
}
#pragma clang module contents
#pragma clang module begin B.X
namespace N {
  class Friendly {
    friend class X;
  };
}
#pragma clang module end
#pragma clang module begin B.Y
namespace N {
  class X;
}
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import A
#pragma clang module import B.X
using N::X;
X *p;

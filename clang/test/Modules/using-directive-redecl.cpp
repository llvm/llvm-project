// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -verify %s
// expected-no-diagnostics
#pragma clang module build M
module M { module TDFNodes {} module TDFInterface {} }
#pragma clang module contents
  // TDFNodes
  #pragma clang module begin M.TDFNodes
  namespace Detail {
     namespace TDF {
        class TLoopManager {};
     }
  }
  namespace Internal {
     namespace TDF {
        using namespace Detail::TDF;
     }
  }
  #pragma clang module end

  // TDFInterface
  #pragma clang module begin M.TDFInterface
    #pragma clang module import M.TDFNodes
      namespace Internal {
        namespace TDF {
          using namespace Detail::TDF;
        }
      }
  #pragma clang module end

#pragma clang module endbuild

#pragma clang module import M.TDFNodes
namespace Internal {
  namespace TDF {
    TLoopManager * use;
  }
}

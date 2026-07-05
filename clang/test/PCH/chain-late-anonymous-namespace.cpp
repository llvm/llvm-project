// no PCH
// RUN: %clang_cc1 -include %s -include %s -fsyntax-only %s
// with PCH
// RUN: %clang_cc1 -chain-include %s -chain-include %s -fsyntax-only %s
// with PCH, with modules enabled
// RUN: %clang_cc1 -chain-include %s -chain-include %s -fsyntax-only -fmodules %s
#if !defined(PASS1)
#define PASS1

namespace ns {}
namespace os {}

#elif !defined(PASS2)
#define PASS2

namespace ns {
  namespace {
    extern int x;
  }
}

namespace {
  extern int y;
}
namespace {
}

namespace os {
  extern "C" {
    namespace {
      extern int z;
    }
  }
}

#else

namespace ns {
  namespace {
    int x;
  }
  void test() {
    (void)x;
  }
}

namespace {
  int y;
}
void test() {
  (void)y;
}

namespace os {
  namespace {
    int z;
  }
  void test() {
    (void)z;
  }
}

#endif

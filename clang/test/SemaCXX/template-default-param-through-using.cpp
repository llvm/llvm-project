// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
namespace llvm {
  template<typename T > struct StringSet;
  template<int I > struct Int;
  template <typename Inner, template <typename> class Outer>
    struct TemplTempl;
}

namespace lld {
  using llvm::StringSet;
  using llvm::Int;
  using llvm::TemplTempl;
};

namespace llvm {
  template<typename T > struct StringSet;
}

template<typename T> struct Temp{};

namespace llvm {
  template<typename T = int> struct StringSet{};
  template<int I = 5> struct Int{};
  template <typename Inner, template <typename> class Outer = Temp>
    struct TemplTempl{};
};

namespace lld {
  StringSet<> s;
  Int<> i;
  TemplTempl<int> tt;
}

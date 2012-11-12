// RUN: %clang_cc1 -ast-print %s 2>&1 | grep "N::M::X<INT>::value"
namespace N {
  namespace M {
    template<typename T>
    struct X {
      enum { value };
    };
  }
}

typedef int INT;

int test() {
  return N::M::X<INT>::value;
}

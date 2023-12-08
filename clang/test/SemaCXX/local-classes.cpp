// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace PR6382 {
  int foo()
  {
    goto error;
    {
      struct BitPacker {
        BitPacker() {}
      };
      BitPacker packer;
    }

  error:
    return -1;
  }
}

namespace PR6383 {
  void test (bool gross)
  {
    struct compare_and_set
    {
      void operator() (const bool inner, const bool gross = false)
      {
        // the code
      }
    } compare_and_set2;

    compare_and_set2 (false, gross);
  }
}

namespace Templates {
  template<int Value>
  void f() {
    struct Inner {
      static int getValue() { return Value; }
    };
  }
}

namespace PR25627_dont_odr_use_local_consts {
  template<int> struct X { X(); X(int); };
  
  void foo() {
    const int N = 10;
  
    struct Local {
      void f(X<N> = X<N>()) {} // OK
    }; 
  }
}

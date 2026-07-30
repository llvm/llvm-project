// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace test0 {
  namespace ns0 {
    class tag;
    int tag();
  }

  namespace ns1 {
    using ns0::tag;
  }

  namespace ns2 {
    using ns0::tag;
  }

  using ns1::tag;
  using ns2::tag;
}

// PR 5752
namespace test1 {
  namespace ns {
    void foo();
  }

  using ns::foo;
  void foo(int);

  namespace ns {
    using test1::foo;
  }
}

// PR 14768
namespace PR14768 {
  template<typename eT> class Mat;
  template<typename eT> class Col : public Mat<eT>   {
    using Mat<eT>::operator();
    using Col<eT>::operator();
    void operator() ();
  };
}

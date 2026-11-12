// RUN: %clang_cc1 -std=c++11 %s -verify 
// expected-no-diagnostics

namespace PR13003 {
  struct void_type
  {
    template <typename Arg0, typename... Args>
    void_type(Arg0&&, Args&&...) { }
  };

  struct void_type2
  {
    template <typename... Args>
    void_type2(Args&&...) { }
  };
  
  struct atom { };
  
  void_type v1 = atom();
  void_type2 v2 = atom();
}


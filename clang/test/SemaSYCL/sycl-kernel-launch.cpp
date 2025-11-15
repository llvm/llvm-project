// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify=host,expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-device -verify=device,expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify=host,expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-device -verify=device,expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify=host,expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -fsycl-is-device -verify=device,expected %s

// Test overload resolution for implicit calls to sycl_kernel_launch<KN>(...)
// synthesized for functions declared with the sycl_kernel_entry_point
// attribute.

////////////////////////////////////////////////////////////////////////////////
// Valid declarations.
////////////////////////////////////////////////////////////////////////////////

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct KN;

// A generic kernel object type.
template<int, int = 0>
struct KT {
  void operator()() const;
};


// sycl_kernel_launch as function template at namespace scope.
namespace ok1 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  [[clang::sycl_kernel_entry_point(KN<1>)]]
  void skep(KT<1> k) {
    k();
  }
}

// sycl_kernel_launch as function template at namespace scope with default
// template arguments and default function arguments..
namespace ok2 {
  template<typename KN, typename T = int>
  void sycl_kernel_launch(const char *, KT<2>, T = 2);
  [[clang::sycl_kernel_entry_point(KN<2>)]]
  void skep(KT<2> k) {
    k();
  }
}

// sycl_kernel_launch as overload set.
namespace ok3 {
  template<typename KN>
  void sycl_kernel_launch(const char *);
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  [[clang::sycl_kernel_entry_point(KN<3>)]]
  void skep(KT<3> k) {
    k();
  }
}

// sycl_kernel_launch as static member function template.
namespace ok4 {
  struct handler {
  private:
    template<typename KN, typename... Ts>
    static void sycl_kernel_launch(const char *, Ts...);
  public:
    [[clang::sycl_kernel_entry_point(KN<4,0>)]]
    static void skep(KT<4,0> k) {
      k();
    }
    [[clang::sycl_kernel_entry_point(KN<4,1>)]]
    void skep(KT<4,1> k) {
      k();
    }
  };
}

// sycl_kernel_launch as non-static member function template.
namespace ok5 {
  struct handler {
  private:
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
  public:
    [[clang::sycl_kernel_entry_point(KN<5>)]]
    void skep(KT<5> k) {
      k();
    }
  };
}

#if __cplusplus >= 202302L
// sycl_kernel_launch as non-static member function template with explicit
// object parameter.
namespace ok6 {
  struct handler {
  private:
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(this handler self, const char *, Ts...);
  public:
    [[clang::sycl_kernel_entry_point(KN<6>)]]
    void skep(KT<6> k) {
      k();
    }
  };
}
#endif

// sycl_kernel_launch as variable template.
namespace ok7 {
  template<typename KN>
  struct launcher {
    template<typename... Ts>
    void operator()(const char *, Ts...);
  };
  template<typename KN>
  launcher<KN> sycl_kernel_launch;
  [[clang::sycl_kernel_entry_point(KN<7>)]]
  void skep(KT<7> k) {
    k();
  }
}

#if __cplusplus >= 202302L
// sycl_kernel_launch as variable template with static call operator template.
namespace ok8 {
  template<typename KN>
  struct launcher {
    template<typename... Ts>
    static void operator()(const char *, Ts...);
  };
  template<typename KN>
  launcher<KN> sycl_kernel_launch;
  [[clang::sycl_kernel_entry_point(KN<8>)]]
  void skep(KT<8> k) {
    k();
  }
}
#endif

#if __cplusplus >= 202302L
// sycl_kernel_launch as variable template with call operator template with
// explicit object parameter.
namespace ok9 {
  template<typename KN>
  struct launcher {
    template<typename... Ts>
    void operator()(this launcher self, const char *, Ts...);
  };
  template<typename KN>
  launcher<KN> sycl_kernel_launch;
  [[clang::sycl_kernel_entry_point(KN<9>)]]
  void skep(KT<9> k) {
    k();
  }
}
#endif

// sycl_kernel_launch as base class non-static member function template.
namespace ok10 {
  template<typename Derived>
  struct base_handler {
  protected:
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
  };
  struct handler : protected base_handler<handler> {
  public:
    [[clang::sycl_kernel_entry_point(KN<10>)]]
    void skep(KT<10> k) {
      k();
    }
  };
}

// sycl_kernel_launch with non-reference parameters.
namespace ok11 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  struct move_only {
    move_only(move_only&&) = default;
  };
  [[clang::sycl_kernel_entry_point(KN<11>)]]
  void skep(KT<11> k, move_only) {
    k();
  }
}

// sycl_kernel_launch with forward reference parameters.
namespace ok12 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts &&...);
  struct non_copyable {
    non_copyable(const non_copyable&) = delete;
  };
  struct non_moveable {
    non_moveable(non_moveable&&) = delete;
  };
  struct move_only {
    move_only(move_only&&) = default;
  };
  [[clang::sycl_kernel_entry_point(KN<12>)]]
  void skep(KT<12> k, non_copyable, non_moveable, move_only) {
    k();
  }
}


////////////////////////////////////////////////////////////////////////////////
// Invalid declarations.
////////////////////////////////////////////////////////////////////////////////

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct BADKN;

// A generic kernel object type.
template<int, int = 0>
struct BADKT {
  void operator()() const;
};

// Undeclared sycl_kernel_launch identifier.
namespace bad1 {
  // host-error@+5 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
  // device-warning@+4 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
  // expected-note@+3 {{define 'sycl_kernel_launch' function template to fix this problem}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  template void skep<BADKN<1>>(BADKT<1>);
}

// No matching function for call to sycl_kernel_launch; not a template.
namespace bad2 {
  // expected-note@+1 {{declared as a non-template here}}
  void sycl_kernel_launch(const char *, BADKT<2>);
  // expected-error@+5 {{'sycl_kernel_launch' does not refer to a template}}
  // host-error@+4 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
  // device-warning@+3 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
  // expected-note@+2 {{define 'sycl_kernel_launch' function template to fix this problem}}
  [[clang::sycl_kernel_entry_point(BADKN<2>)]]
  void skep(BADKT<2> k) {
    k();
  }
}

// No matching function for call to sycl_kernel_launch; not enough arguments.
namespace bad3 {
  // expected-note@+2 {{candidate function template not viable: requires 2 arguments, but 1 was provided}}
  template<typename KN, typename KT>
  void sycl_kernel_launch(const char *, KT);
  // expected-error@+3 {{no matching function for call to 'sycl_kernel_launch'}}
  template<typename KN>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep() {}
  // expected-note@+1 {{in instantiation of function template specialization 'bad3::skep<BADKN<3>>' requested here}}
  template void skep<BADKN<3>>();
}

// No matching function for call to sycl_kernel_launch; too many arguments.
namespace bad4 {
  // expected-note@+2 {{candidate function template not viable: requires 2 arguments, but 3 were provided}}
  template<typename KN, typename KT>
  void sycl_kernel_launch(const char *, KT);
  // expected-error@+3 {{no matching function for call to 'sycl_kernel_launch'}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k, int i) {
    k();
  }
  // expected-note@+1 {{in instantiation of function template specialization 'bad4::skep<BADKN<4>, BADKT<4>>' requested here}}
  template void skep<BADKN<4>>(BADKT<4>, int);
}

// No matching function for call to sycl_kernel_launch; mismatched function parameter type.
namespace bad5 {
  // expected-note@+2 {{candidate function template not viable: no known conversion from 'const char[21]' to 'int' for 1st argument}}
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(int, Ts...);
  // expected-error@+3 {{no matching function for call to 'sycl_kernel_launch'}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  // expected-note@+1 {{in instantiation of function template specialization 'bad5::skep<BADKN<5>, BADKT<5>>' requested here}}
  template void skep<BADKN<5>>(BADKT<5>);
}

// No matching function for call to sycl_kernel_launch; mismatched template parameter kind.
namespace bad6 {
  // expected-note@+2 {{candidate template ignored: invalid explicitly-specified argument for 1st template parameter}}
  template<int, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  // expected-error@+2 {{no matching function for call to 'sycl_kernel_launch'}}
  [[clang::sycl_kernel_entry_point(BADKN<6>)]]
  void skep(BADKT<6> k) {
    k();
  }
}

// No matching function for call to sycl_kernel_launch object; mismatched function parameter type.
namespace bad7 {
  template<typename KN>
  struct launcher {
    // expected-note@+2 {{candidate function template not viable: no known conversion from 'const char[21]' to 'int' for 1st argument}}
    template<typename... Ts>
    void operator()(int, Ts...);
  };
  template<typename KN>
  launcher<KN> sycl_kernel_launch;
  // expected-error@+3 {{no matching function for call to object of type 'launcher<BADKN<7, 0>>'}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  // expected-note@+1 {{in instantiation of function template specialization 'bad7::skep<BADKN<7>, BADKT<7>>' requested here}}
  template void skep<BADKN<7>>(BADKT<7>);
}

// No matching function for call to sycl_kernel_launch object; mismatched template parameter kind.
namespace bad8 {
  template<int KN>
  struct launcher {
    template<typename... Ts>
    void operator()(int, Ts...);
  };
  // expected-note@+1 {{template parameter is declared here}}
  template<int KN>
  launcher<KN> sycl_kernel_launch;
  // expected-error@+3 {{template argument for non-type template parameter must be an expression}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  template void skep<BADKN<8>>(BADKT<8>);
}

// sycl_kernel_launch as variable template with private call operator template.
namespace bad9 {
  template<typename KN>
  struct launcher {
  private:
    // expected-note@+2 {{declared private here}}
    template<typename... Ts>
    void operator()(const char *, Ts...);
  };
  template<typename KN>
  launcher<KN> sycl_kernel_launch;
  // expected-error@+2 {{'operator()' is a private member of 'bad9::launcher<BADKN<9>>'}}
  [[clang::sycl_kernel_entry_point(BADKN<9>)]]
  void skep(BADKT<9> k) {
    k();
  }
}

// Ambiguous reference to sycl_kernel_launch.
namespace bad10 {
  inline namespace in1 {
    // expected-note@+2 {{candidate found by name lookup is 'bad10::in1::sycl_kernel_launch'}}
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
  }
  inline namespace in2 {
    template<typename KN>
    struct launcher {
      template<typename KT, typename... Ts>
      void operator()(const char *, Ts...);
    };
    // expected-note@+2 {{candidate found by name lookup is 'bad10::in2::sycl_kernel_launch'}}
    template<typename KN>
    launcher<KN> sycl_kernel_launch;
  }
  // expected-error@+6 {{reference to 'sycl_kernel_launch' is ambiguous}}
  // host-error@+5 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
  // device-warning@+4 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
  // expected-note@+3 {{define 'sycl_kernel_launch' function template to fix this problem}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  template void skep<BADKN<10>>(BADKT<10>);
}

// Ambiguous call to sycl_kernel_launch.
namespace bad11 {
  // expected-note@+2 {{candidate function [with KN = BADKN<11>, KT = BADKT<11>]}}
  template<typename KN, typename KT>
  void sycl_kernel_launch(const char *, KT, signed char);
  // expected-note@+2 {{candidate function [with KN = BADKN<11>, KT = BADKT<11>]}}
  template<typename KN, typename KT>
  void sycl_kernel_launch(const char *, KT, unsigned char);
  // expected-error@+2 {{call to 'sycl_kernel_launch' is ambiguous}}
  [[clang::sycl_kernel_entry_point(BADKN<11>)]]
  void skep(BADKT<11> k, int i) {
    k();
  }
}

// Call to member sycl_kernel_launch from non-static member.
namespace bad12 {
  struct S {
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
    // expected-error@+2 {{call to non-static member function without an object argument}}
    [[clang::sycl_kernel_entry_point(BADKN<12>)]]
    static void skep(BADKT<12> k) {
      k();
    }
  };
}

// sycl_kernel_launch as dependent base class non-static member function
// template.
namespace bad13 {
  template<typename Derived>
  struct base_handler {
  protected:
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
  };
  template<int N>
  struct handler : protected base_handler<handler<N>> {
    // Lookup for sycl_kernel_launch fails because lookup in dependent base
    // classes requires explicit qualification.
    // host-error@+4 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
    // device-warning@+3 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
    // expected-note@+2 {{define 'sycl_kernel_launch' function template to fix this problem}}
    [[clang::sycl_kernel_entry_point(BADKN<13>)]]
    void skep(BADKT<13> k) {
      k();
    }
  };
  template void handler<13>::skep(BADKT<13>);
}

// sycl_kernel_launch with non-reference parameters and non-moveable arguments.
namespace bad14 {
  // expected-note@+2 2 {{passing argument to parameter here}}
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  struct non_copyable {
    // expected-note@+1 {{'non_copyable' has been explicitly marked deleted here}}
    non_copyable(const non_copyable&) = delete;
  };
  // expected-error@+2 {{call to deleted constructor of 'bad14::non_copyable'}}
  [[clang::sycl_kernel_entry_point(BADKN<14,0>)]]
  void skep(BADKT<14,0> k, non_copyable) {
    k();
  }
  struct non_moveable {
    // expected-note@+1 {{'non_moveable' has been explicitly marked deleted here}}
    non_moveable(non_moveable&&) = delete;
  };
  // expected-error@+2 {{call to deleted constructor of 'bad14::non_moveable'}}
  [[clang::sycl_kernel_entry_point(BADKN<14,1>)]]
  void skep(BADKT<14,1> k, non_moveable) {
    k();
  }
}

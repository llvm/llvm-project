// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -fsycl-is-device -verify %s

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

// ADL for sycl_kernel_launch.
namespace ok13 {
  template<typename KN, typename KT, typename T>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k, T t) {
    k();
  }
  namespace nested {
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
    struct S13 {};
  }
  template void skep<KN<13>>(KT<13>, nested::S13);
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


// Undeclared sycl_kernel_launch identifier from non-template function.
namespace bad1 {
  // expected-error@+4 {{use of undeclared identifier 'sycl_kernel_launch'}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<1>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<1>') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<1>)]]
  void skep(BADKT<1> k) {
    k();
  }
}

// Undeclared sycl_kernel_launch identifier from function template.
namespace bad2 {
  // expected-error@+5 {{use of undeclared identifier 'sycl_kernel_launch'}}
  // expected-note@+3 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+2 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<2>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<2>') required here}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  // expected-note@+1 {{in instantiation of function template specialization 'bad2::skep<BADKN<2>, BADKT<2>>' requested here}}
  template void skep<BADKN<2>>(BADKT<2>);
}

// No matching function for call to sycl_kernel_launch; not a template.
namespace bad3 {
  // expected-note@+1 {{declared as a non-template here}}
  void sycl_kernel_launch(const char *, BADKT<3>);
  // expected-error@+4 {{'sycl_kernel_launch' does not refer to a template}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<3>' required here}}
  [[clang::sycl_kernel_entry_point(BADKN<3>)]]
  void skep(BADKT<3> k) {
    k();
  }
}

// No matching function for call to sycl_kernel_launch; not enough arguments.
namespace bad4 {
  // expected-note@+2 {{candidate function template not viable: requires 2 arguments, but 1 was provided}}
  template<typename KN, typename KT>
  void sycl_kernel_launch(const char *, KT);
  // expected-error@+5 {{no matching function for call to 'sycl_kernel_launch'}}
  // expected-note@+3 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+2 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<4>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]') required here}}
  template<typename KN>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep() {}
  // expected-note@+1 {{in instantiation of function template specialization 'bad4::skep<BADKN<4>>' requested here}}
  template void skep<BADKN<4>>();
}

// No matching function for call to sycl_kernel_launch; too many arguments.
namespace bad5 {
  // expected-note@+2 {{candidate function template not viable: requires 2 arguments, but 3 were provided}}
  template<typename KN, typename KT>
  void sycl_kernel_launch(const char *, KT);
  // expected-error@+5 {{no matching function for call to 'sycl_kernel_launch'}}
  // expected-note@+3 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+2 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<5>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<5>', xvalue of type 'int') required here}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k, int i) {
    k();
  }
  // expected-note@+1 {{in instantiation of function template specialization 'bad5::skep<BADKN<5>, BADKT<5>>' requested here}}
  template void skep<BADKN<5>>(BADKT<5>, int);
}

// No matching function for call to sycl_kernel_launch; mismatched function parameter type.
namespace bad6 {
  // expected-note-re@+2 {{candidate function template not viable: no known conversion from 'const char[{{[0-9]*}}]' to 'int' for 1st argument}}
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(int, Ts...);
  // expected-error@+5 {{no matching function for call to 'sycl_kernel_launch'}}
  // expected-note@+3 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+2 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<6>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<6>') required here}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  // expected-note@+1 {{in instantiation of function template specialization 'bad6::skep<BADKN<6>, BADKT<6>>' requested here}}
  template void skep<BADKN<6>>(BADKT<6>);
}

// No matching function for call to sycl_kernel_launch; mismatched template parameter kind.
namespace bad7 {
  // expected-note@+2 {{candidate template ignored: invalid explicitly-specified argument for 1st template parameter}}
  template<int, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  // expected-error@+4 {{no matching function for call to 'sycl_kernel_launch'}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<7>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<7>') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<7>)]]
  void skep(BADKT<7> k) {
    k();
  }
}

// No matching function for call to sycl_kernel_launch; substitution failure.
namespace bad8 {
  // expected-note@+2 {{candidate template ignored: substitution failure [with KN = BADKN<8>, KT = BADKT<8>]: no type named 'no_such_type' in 'BADKT<8>'}}
  template<typename KN, typename KT, typename T = typename KT::no_such_type>
  void sycl_kernel_launch(const char *, KT);
  // expected-error@+4 {{no matching function for call to 'sycl_kernel_launch'}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<8>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<8>') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<8>)]]
  void skep(BADKT<8> k) {
    k();
  }
}

// No matching function for call to sycl_kernel_launch; deduction failure.
namespace bad9 {
  // expected-note@+2 {{candidate template ignored: couldn't infer template argument 'T'}}
  template<typename KN, typename KT, typename T>
  void sycl_kernel_launch(const char *, KT);
  // expected-error@+4 {{no matching function for call to 'sycl_kernel_launch'}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<9>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<9>') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<9>)]]
  void skep(BADKT<9> k) {
    k();
  }
}

// No matching function for call to sycl_kernel_launch object; mismatched function parameter type.
namespace bad10 {
  template<typename KN>
  struct launcher {
    // expected-note-re@+2 {{candidate function template not viable: no known conversion from 'const char[{{[0-9]*}}]' to 'int' for 1st argument}}
    template<typename... Ts>
    void operator()(int, Ts...);
  };
  template<typename KN>
  launcher<KN> sycl_kernel_launch;
  // expected-error@+5 {{no matching function for call to object of type 'launcher<BADKN<10, 0>>'}}
  // expected-note@+3 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+2 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<10>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<10>') required here}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  // expected-note@+1 {{in instantiation of function template specialization 'bad10::skep<BADKN<10>, BADKT<10>>' requested here}}
  template void skep<BADKN<10>>(BADKT<10>);
}

// No matching function for call to sycl_kernel_launch object; mismatched template parameter kind.
namespace bad11 {
  template<int KN>
  struct launcher {
    template<typename... Ts>
    void operator()(int, Ts...);
  };
  // expected-note@+1 {{template parameter is declared here}}
  template<int KN>
  launcher<KN> sycl_kernel_launch;
  // expected-error@+5 {{template argument for non-type template parameter must be an expression}}
  // expected-note@+3 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note@+2 {{in implicit call to 'sycl_kernel_launch' with template argument 'KN' required here}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  template void skep<BADKN<11>>(BADKT<11>);
}

// sycl_kernel_launch as variable template with private call operator template.
namespace bad12 {
  template<typename KN>
  struct launcher {
  private:
    // expected-note@+2 {{declared private here}}
    template<typename... Ts>
    void operator()(const char *, Ts...);
  };
  template<typename KN>
  launcher<KN> sycl_kernel_launch;
  // expected-error@+4 {{'operator()' is a private member of 'bad12::launcher<BADKN<12>>'}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<12>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<12>') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<12>)]]
  void skep(BADKT<12> k) {
    k();
  }
}

// Ambiguous reference to sycl_kernel_launch.
namespace bad13 {
  inline namespace in1 {
    // expected-note@+2 {{candidate found by name lookup is 'bad13::in1::sycl_kernel_launch'}}
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
  }
  inline namespace in2 {
    template<typename KN>
    struct launcher {
      template<typename KT, typename... Ts>
      void operator()(const char *, Ts...);
    };
    // expected-note@+2 {{candidate found by name lookup is 'bad13::in2::sycl_kernel_launch'}}
    template<typename KN>
    launcher<KN> sycl_kernel_launch;
  }
  // expected-error@+5 {{reference to 'sycl_kernel_launch' is ambiguous}}
  // expected-note@+3 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note@+2 {{in implicit call to 'sycl_kernel_launch' with template argument 'KN' required here}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  template void skep<BADKN<13>>(BADKT<13>);
}

// Ambiguous call to sycl_kernel_launch.
namespace bad14 {
  // expected-note@+2 {{candidate function [with KN = BADKN<14>, KT = BADKT<14>]}}
  template<typename KN, typename KT>
  void sycl_kernel_launch(const char *, KT, signed char);
  // expected-note@+2 {{candidate function [with KN = BADKN<14>, KT = BADKT<14>]}}
  template<typename KN, typename KT>
  void sycl_kernel_launch(const char *, KT, unsigned char);
  // expected-error@+4 {{call to 'sycl_kernel_launch' is ambiguous}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<14>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<14>', xvalue of type 'int') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<14>)]]
  void skep(BADKT<14> k, int i) {
    k();
  }
}

// Call to member sycl_kernel_launch from non-static member.
namespace bad15 {
  struct S {
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
    // expected-error@+4 {{call to non-static member function without an object argument}}
    // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
    // expected-note@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<15>' required here}}
    [[clang::sycl_kernel_entry_point(BADKN<15>)]]
    static void skep(BADKT<15> k) {
      k();
    }
  };
}

// sycl_kernel_launch as dependent base class non-static member function
// template.
namespace bad16 {
  template<typename Derived>
  struct base_handler {
  protected:
    // expected-note@+2 {{member is declared here}}
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
  };
  template<int N>
  struct handler : protected base_handler<handler<N>> {
    // Lookup for sycl_kernel_launch fails because lookup in dependent base
    // classes requires explicit qualification.
    // expected-error@+4 {{explicit qualification required to use member 'sycl_kernel_launch' from dependent base class}}
    // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
    // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<16>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<16>') required here}}
    [[clang::sycl_kernel_entry_point(BADKN<16>)]]
    void skep(BADKT<16> k) {
      k();
    }
  };
  // expected-note@+1 {{in instantiation of member function 'bad16::handler<16>::skep' requested here}}
  template void handler<16>::skep(BADKT<16>);
}

// sycl_kernel_launch with non-reference parameters and non-moveable arguments.
namespace bad17 {
  // expected-note@+2 2 {{passing argument to parameter here}}
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  struct non_copyable {
    // expected-note@+1 {{'non_copyable' has been explicitly marked deleted here}}
    non_copyable(const non_copyable&) = delete;
  };
  // expected-error@+4 {{call to deleted constructor of 'bad17::non_copyable'}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<17, 0>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<17, 0>', xvalue of type 'non_copyable') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<17,0>)]]
  void skep(BADKT<17,0> k, non_copyable) {
    k();
  }
  struct non_moveable {
    // expected-note@+1 {{'non_moveable' has been explicitly marked deleted here}}
    non_moveable(non_moveable&&) = delete;
  };
  // expected-error@+4 {{call to deleted constructor of 'bad17::non_moveable'}}
  // expected-note@+2 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<17, 1>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<17, 1>', xvalue of type 'non_moveable') required here}}
  [[clang::sycl_kernel_entry_point(BADKN<17,1>)]]
  void skep(BADKT<17,1> k, non_moveable) {
    k();
  }
}

// sycl_kernel_launch declared after use and not found by ADL.
namespace bad18 {
  // expected-error@+5 {{call to function 'sycl_kernel_launch' that is neither visible in the template definition nor found by argument-dependent lookup}}
  // expected-note@+3 {{this error is due to a defect in SYCL runtime header files; please report this problem to your SYCL runtime provider}}
  // expected-note-re@+2 {{in implicit call to 'sycl_kernel_launch' with template argument 'BADKN<18>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'BADKT<18>') required here}}
  template<typename KN, typename KT>
  [[clang::sycl_kernel_entry_point(KN)]]
  void skep(KT k) {
    k();
  }
  // expected-note@+2 {{'sycl_kernel_launch' should be declared prior to the call site or in the global namespace}}
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(Ts...) {}
  // expected-note@+1 {{in instantiation of function template specialization 'bad18::skep<BADKN<18>, BADKT<18>>' requested here}}
  template void skep<BADKN<18>>(BADKT<18>);
}

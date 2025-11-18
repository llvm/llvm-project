// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 -fsyntax-only -fsycl-is-host -fms-compatibility -fcxx-exceptions -verify=host,expected %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 -fsyntax-only -fsycl-is-device -fms-compatibility -verify=device,expected %s

// Test Microsoft extensions for lookup of a sycl_kernel_launch member template
// in a dependent base class.

////////////////////////////////////////////////////////////////////////////////
// Valid declarations.
////////////////////////////////////////////////////////////////////////////////

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

// A generic kernel object type.
template<int>
struct KT {
  void operator()() const;
};


namespace ok1 {
  template<typename Derived>
  struct base_handler {
  protected:
    // expected-note@+2 {{must qualify identifier to find this declaration in dependent base class}}
    template<typename KN, typename... Ts>
    void sycl_kernel_launch(const char *, Ts...);
  };
  template<int N>
  struct handler : protected base_handler<handler<N>> {
    // A warning is issued because, in standard C++, unqualified lookup for
    // sycl_kernel_launch would not consider dependent base classes. Such
    // lookups are allowed as a Microsoft compatible extension.
    // expected-warning@+3 {{use of member 'sycl_kernel_launch' found via unqualified lookup into dependent bases of class templates is a Microsoft extension}}
    // expected-note-re@+1 {{in implicit call to 'sycl_kernel_launch' with template argument 'KN<1>' and function arguments (lvalue of type 'const char[{{[0-9]*}}]', xvalue of type 'KT<1>') required here}}
    [[clang::sycl_kernel_entry_point(KN<1>)]]
    void skep(KT<1> k) {
      k();
    }
  };
  // expected-note@+1 {{in instantiation of member function 'ok1::handler<1>::skep' requested here}}
  template void handler<1>::skep(KT<1>);
}

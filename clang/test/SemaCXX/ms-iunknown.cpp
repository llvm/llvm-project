// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s 

struct __declspec(uuid("00000000-0000-0000-C000-000000000046")) IUnknown {
  void foo();
};
struct IPropertyPageBase : public IUnknown {};
struct IPropertyPage : public IPropertyPageBase {};
__interface ISfFileIOPropertyPage : public IPropertyPage {};


namespace NS {
  struct __declspec(uuid("00000000-0000-0000-C000-000000000046")) IUnknown {};
  // expected-error@+1 {{interface type cannot inherit from}}
  __interface IPropertyPageBase : public IUnknown {}; 
}
// expected-error@+1 {{interface type cannot inherit from}}
__interface IPropertyPageBase2 : public NS::IUnknown {}; 

__interface temp_iface {};
struct bad_base : temp_iface {};
// expected-error@+1 {{interface type cannot inherit from}}
__interface bad_inherit : public bad_base{};

struct mult_inher_base : temp_iface, IUnknown {};
// expected-error@+1 {{interface type cannot inherit from}}
__interface bad_inherit2 : public mult_inher_base{};

struct PageBase : public IUnknown {};
struct Page3 : public PageBase {};
struct Page4 : public PageBase {};
__interface PropertyPage : public Page4 {};

struct Page5 : public Page3, Page4{};
// expected-error@+1 {{interface type cannot inherit from}}
__interface PropertyPage2 : public Page5 {}; 

__interface IF1 {};
__interface PP : IUnknown, IF1{}; 
__interface PP2 : PP, Page3, Page4{};

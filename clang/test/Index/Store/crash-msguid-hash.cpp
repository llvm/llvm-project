// RUN: rm -rf %t
// RUN: %clang_cc1 %s -std=c++1z -index-store-path %t/idx -fms-extensions
// Should not crash.

class VirtualBase
{
public:
   VirtualBase() noexcept {}
   virtual ~VirtualBase() noexcept {}
};

template< class MostDerivedInterface, const _GUID& MostDerivedInterfaceIID = __uuidof( MostDerivedInterface )  >
class TGuidHolder : public VirtualBase
{
};

struct  __declspec(uuid("12345678-1234-5678-ABCD-12345678ABCD")) GUIDInterfaceClass {};

class Widget : public TGuidHolder< GUIDInterfaceClass >
{
public:
   Widget();
   ~Widget() noexcept {}
};


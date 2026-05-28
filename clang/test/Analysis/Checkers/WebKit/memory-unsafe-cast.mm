// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.MemoryUnsafeCastChecker -verify %s

#include "mock-types.h"
#include "objc-mock-types.h"

@interface BaseClass : NSObject
@end

@interface DerivedClass : BaseClass
-(void)testCasts:(BaseClass*)base;
@end

@implementation DerivedClass
-(void)testCasts:(BaseClass*)base {
  DerivedClass *derived = (DerivedClass*)base;
  // expected-warning@-1{{Unsafe cast from base type 'BaseClass' to derived type 'DerivedClass'}}
  DerivedClass *derived_static = static_cast<DerivedClass*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'BaseClass' to derived type 'DerivedClass'}}
  DerivedClass *derived_reinterpret = reinterpret_cast<DerivedClass*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'BaseClass' to derived type 'DerivedClass'}}
  base = (BaseClass*)derived;  // no warning
  base = (BaseClass*)base;  // no warning
}
@end

template <typename T>
class WrappedObject
{
public:
  T get() const { return mMetalObject; }
  T mMetalObject = nullptr;
};

@protocol MTLCommandEncoder
@end
@protocol MTLRenderCommandEncoder
@end
class CommandEncoder : public WrappedObject<id<MTLCommandEncoder>> { };

class RenderCommandEncoder final : public CommandEncoder
{
private:
    id<MTLRenderCommandEncoder> get()
    {
        return static_cast<id<MTLRenderCommandEncoder>>(CommandEncoder::get());
    }
};

@interface Class1
@end

@interface Class2
@end

void testUnrelated(Class1 *c1) {
  Class2 *c2 = (Class2*)c1;
  // expected-warning@-1{{Unsafe cast from type 'Class1' to an unrelated type 'Class2'}}
  Class1 *c1_same = reinterpret_cast<Class1*>(c1); // no warning
}

struct Base : RefCountable { virtual ~Base() {} };
struct Derived : Base { int extra; };

void* returnCast(Base* base) {
  return static_cast<void *>(base);
}

Derived* fnArgCast(void* base) {
  return static_cast<Derived*>(base);
}

void fn_cast_01(Base* base) {
  auto* d1 = static_cast<Derived*>(base);
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  auto* d2 = static_cast<Derived*>(static_cast<void*>(base));
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  auto* d3 = static_cast<Derived*>(returnCast(base));
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  auto* d4 = fnArgCast(static_cast<void*>(base));
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
  fnArgCast(static_cast<void*>(base));
  // expected-warning@-1{{Unsafe cast from base type 'Base' to derived type 'Derived'}}
}

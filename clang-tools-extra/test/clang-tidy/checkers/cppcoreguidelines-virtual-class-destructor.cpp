// RUN: %check_clang_tidy %s cppcoreguidelines-virtual-class-destructor %t -- --fix-notes

// CHECK-MESSAGES: :[[@LINE+4]]:8: warning: destructor of 'PrivateVirtualBaseStruct' is private and prevents using the type [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+3]]:8: note: make it public and virtual
// CHECK-MESSAGES: :[[@LINE+2]]:8: note: make it protected
// As we have 2 conflicting fixes in notes, no fix is applied.
struct PrivateVirtualBaseStruct {
  virtual void f();

private:
  virtual ~PrivateVirtualBaseStruct() {}
};

struct PublicVirtualBaseStruct { // OK
  virtual void f();
  virtual ~PublicVirtualBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: destructor of 'ProtectedVirtualBaseStruct' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:8: note: make it protected and non-virtual
struct ProtectedVirtualBaseStruct {
  virtual void f();

protected:
  virtual ~ProtectedVirtualBaseStruct() {}
  // CHECK-FIXES: ~ProtectedVirtualBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: destructor of 'ProtectedVirtualDefaultBaseStruct' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:8: note: make it protected and non-virtual
struct ProtectedVirtualDefaultBaseStruct {
  virtual void f();

protected:
  virtual ~ProtectedVirtualDefaultBaseStruct() = default;
  // CHECK-FIXES: ~ProtectedVirtualDefaultBaseStruct() = default;
};

// CHECK-MESSAGES: :[[@LINE+4]]:8: warning: destructor of 'PrivateNonVirtualBaseStruct' is private and prevents using the type [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+3]]:8: note: make it public and virtual
// CHECK-MESSAGES: :[[@LINE+2]]:8: note: make it protected
// As we have 2 conflicting fixes in notes, no fix is applied.
struct PrivateNonVirtualBaseStruct {
  virtual void f();

private:
  ~PrivateNonVirtualBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: destructor of 'PublicNonVirtualBaseStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:8: note: make it public and virtual
struct PublicNonVirtualBaseStruct {
  virtual void f();
  ~PublicNonVirtualBaseStruct() {}
  // CHECK-FIXES: virtual ~PublicNonVirtualBaseStruct() {}
};

struct PublicNonVirtualNonBaseStruct { // OK according to C.35, since this struct does not have any virtual methods.
  void f();
  ~PublicNonVirtualNonBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+4]]:8: warning: destructor of 'PublicImplicitNonVirtualBaseStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+3]]:8: note: make it public and virtual
// CHECK-FIXES: struct PublicImplicitNonVirtualBaseStruct {
// CHECK-FIXES-NEXT: virtual ~PublicImplicitNonVirtualBaseStruct() = default;
struct PublicImplicitNonVirtualBaseStruct {
  virtual void f();
};

// CHECK-MESSAGES: :[[@LINE+5]]:8: warning: destructor of 'PublicASImplicitNonVirtualBaseStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+4]]:8: note: make it public and virtual
// CHECK-FIXES: struct PublicASImplicitNonVirtualBaseStruct {
// CHECK-FIXES-NEXT: virtual ~PublicASImplicitNonVirtualBaseStruct() = default;
// CHECK-FIXES-NEXT: private:
struct PublicASImplicitNonVirtualBaseStruct {
private:
  virtual void f();
};

struct ProtectedNonVirtualBaseStruct { // OK
  virtual void f();

protected:
  ~ProtectedNonVirtualBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+4]]:7: warning: destructor of 'PrivateVirtualBaseClass' is private and prevents using the type [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+3]]:7: note: make it public and virtual
// CHECK-MESSAGES: :[[@LINE+2]]:7: note: make it protected
// As we have 2 conflicting fixes in notes, no fix is applied.
class PrivateVirtualBaseClass {
  virtual void f();
  virtual ~PrivateVirtualBaseClass() {}
};

class PublicVirtualBaseClass { // OK
  virtual void f();

public:
  virtual ~PublicVirtualBaseClass() {}
};

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: destructor of 'ProtectedVirtualBaseClass' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:7: note: make it protected and non-virtual
class ProtectedVirtualBaseClass {
  virtual void f();

protected:
  virtual ~ProtectedVirtualBaseClass() {}
  // CHECK-FIXES: ~ProtectedVirtualBaseClass() {}
};

// CHECK-MESSAGES: :[[@LINE+5]]:7: warning: destructor of 'PublicImplicitNonVirtualBaseClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+4]]:7: note: make it public and virtual
// CHECK-FIXES: public:
// CHECK-FIXES-NEXT: virtual ~PublicImplicitNonVirtualBaseClass() = default;
// CHECK-FIXES-NEXT: };
class PublicImplicitNonVirtualBaseClass {
  virtual void f();
};

// CHECK-MESSAGES: :[[@LINE+6]]:7: warning: destructor of 'PublicASImplicitNonVirtualBaseClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+5]]:7: note: make it public and virtual
// CHECK-FIXES: public:
// CHECK-FIXES-NEXT: virtual ~PublicASImplicitNonVirtualBaseClass() = default;
// CHECK-FIXES-NEXT: int foo = 42;
// CHECK-FIXES-NEXT: };
class PublicASImplicitNonVirtualBaseClass {
  virtual void f();

public:
  int foo = 42;
};

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: destructor of 'PublicNonVirtualBaseClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:7: note: make it public and virtual
class PublicNonVirtualBaseClass {
  virtual void f();

public:
  ~PublicNonVirtualBaseClass() {}
  // CHECK-FIXES: virtual ~PublicNonVirtualBaseClass() {}
};

class PublicNonVirtualNonBaseClass { // OK accoring to C.35, since this class does not have any virtual methods.
  void f();

public:
  ~PublicNonVirtualNonBaseClass() {}
};

class ProtectedNonVirtualClass { // OK
public:
  virtual void f();

protected:
  ~ProtectedNonVirtualClass() {}
};

// CHECK-MESSAGES: :[[@LINE+7]]:7: warning: destructor of 'OverridingDerivedClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+6]]:7: note: make it public and virtual
// CHECK-FIXES: class OverridingDerivedClass : ProtectedNonVirtualClass {
// CHECK-FIXES-NEXT: public:
// CHECK-FIXES-NEXT: virtual ~OverridingDerivedClass() = default;
// CHECK-FIXES-NEXT: void f() override;
// CHECK-FIXES-NEXT: };
class OverridingDerivedClass : ProtectedNonVirtualClass {
public:
  void f() override; // is implicitly virtual
};

// CHECK-MESSAGES: :[[@LINE+7]]:7: warning: destructor of 'NonOverridingDerivedClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+6]]:7: note: make it public and virtual
// CHECK-FIXES: class NonOverridingDerivedClass : ProtectedNonVirtualClass {
// CHECK-FIXES-NEXT: void m();
// CHECK-FIXES-NEXT: public:
// CHECK-FIXES-NEXT: virtual ~NonOverridingDerivedClass() = default;
// CHECK-FIXES-NEXT: };
class NonOverridingDerivedClass : ProtectedNonVirtualClass {
  void m();
};
// inherits virtual method

// CHECK-MESSAGES: :[[@LINE+6]]:8: warning: destructor of 'OverridingDerivedStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+5]]:8: note: make it public and virtual
// CHECK-FIXES: struct OverridingDerivedStruct : ProtectedNonVirtualBaseStruct {
// CHECK-FIXES-NEXT: virtual ~OverridingDerivedStruct() = default;
// CHECK-FIXES-NEXT: void f() override;
// CHECK-FIXES-NEXT: };
struct OverridingDerivedStruct : ProtectedNonVirtualBaseStruct {
  void f() override; // is implicitly virtual
};

// CHECK-MESSAGES: :[[@LINE+6]]:8: warning: destructor of 'NonOverridingDerivedStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+5]]:8: note: make it public and virtual
// CHECK-FIXES: struct NonOverridingDerivedStruct : ProtectedNonVirtualBaseStruct {
// CHECK-FIXES-NEXT: virtual ~NonOverridingDerivedStruct() = default;
// CHECK-FIXES-NEXT: void m();
// CHECK-FIXES-NEXT: };
struct NonOverridingDerivedStruct : ProtectedNonVirtualBaseStruct {
  void m();
};
// inherits virtual method

namespace Bugzilla_51912 {
// Fixes https://bugs.llvm.org/show_bug.cgi?id=51912

// Forward declarations
// CHECK-MESSAGES-NOT: :[[@LINE+1]]:8: warning: destructor of 'ForwardDeclaredStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
struct ForwardDeclaredStruct;

struct ForwardDeclaredStruct : PublicVirtualBaseStruct {
};

// Normal Template
// CHECK-MESSAGES-NOT: :[[@LINE+2]]:8: warning: destructor of 'TemplatedDerived' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
template <typename T>
struct TemplatedDerived : PublicVirtualBaseStruct {
};

TemplatedDerived<int> InstantiationWithInt;

// Derived from template, base has virtual dtor
// CHECK-MESSAGES-NOT: :[[@LINE+2]]:8: warning: destructor of 'DerivedFromTemplateVirtualBaseStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
template <typename T>
struct DerivedFromTemplateVirtualBaseStruct : T {
  virtual void foo();
};

DerivedFromTemplateVirtualBaseStruct<PublicVirtualBaseStruct> InstantiationWithPublicVirtualBaseStruct;

// Derived from template, base has *not* virtual dtor
// CHECK-MESSAGES: :[[@LINE+7]]:8: warning: destructor of 'DerivedFromTemplateNonVirtualBaseStruct<PublicNonVirtualBaseStruct>' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+6]]:8: note: make it public and virtual
// CHECK-FIXES: struct DerivedFromTemplateNonVirtualBaseStruct : T {
// CHECK-FIXES-NEXT: virtual ~DerivedFromTemplateNonVirtualBaseStruct() = default;
// CHECK-FIXES-NEXT: virtual void foo();
// CHECK-FIXES-NEXT: };
template <typename T>
struct DerivedFromTemplateNonVirtualBaseStruct : T {
  virtual void foo();
};

DerivedFromTemplateNonVirtualBaseStruct<PublicNonVirtualBaseStruct> InstantiationWithPublicNonVirtualBaseStruct;

// Derived from template, base has virtual dtor, to be used in a typedef
// CHECK-MESSAGES-NOT: :[[@LINE+2]]:8: warning: destructor of 'DerivedFromTemplateVirtualBaseStruct2' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
template <typename T>
struct DerivedFromTemplateVirtualBaseStruct2 : T {
  virtual void foo();
};

using DerivedFromTemplateVirtualBaseStruct2Typedef = DerivedFromTemplateVirtualBaseStruct2<PublicVirtualBaseStruct>;
DerivedFromTemplateVirtualBaseStruct2Typedef InstantiationWithPublicVirtualBaseStruct2;

// Derived from template, base has *not* virtual dtor, to be used in a typedef
// CHECK-MESSAGES: :[[@LINE+7]]:8: warning: destructor of 'DerivedFromTemplateNonVirtualBaseStruct2<PublicNonVirtualBaseStruct>' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+6]]:8: note: make it public and virtual
// CHECK-FIXES: struct DerivedFromTemplateNonVirtualBaseStruct2 : T {
// CHECK-FIXES-NEXT: virtual ~DerivedFromTemplateNonVirtualBaseStruct2() = default;
// CHECK-FIXES-NEXT: virtual void foo();
// CHECK-FIXES-NEXT: };
template <typename T>
struct DerivedFromTemplateNonVirtualBaseStruct2 : T {
  virtual void foo();
};

using DerivedFromTemplateNonVirtualBaseStruct2Typedef = DerivedFromTemplateNonVirtualBaseStruct2<PublicNonVirtualBaseStruct>;
DerivedFromTemplateNonVirtualBaseStruct2Typedef InstantiationWithPublicNonVirtualBaseStruct2;

} // namespace Bugzilla_51912

namespace macro_tests {
#define CONCAT(x, y) x##y

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: destructor of 'FooBar1' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:7: note: make it protected and non-virtual
class FooBar1 {
protected:
  CONCAT(vir, tual) CONCAT(~Foo, Bar1()); // no-fixit
};

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: destructor of 'FooBar2' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:7: note: make it protected and non-virtual
class FooBar2 {
protected:
  virtual CONCAT(~Foo, Bar2()); // FIXME: We should have a fixit for this.
};

// CHECK-MESSAGES: :[[@LINE+6]]:7: warning: destructor of 'FooBar3' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+5]]:7: note: make it protected and non-virtual
// CHECK-FIXES:      class FooBar3 {
// CHECK-FIXES-NEXT: protected:
// CHECK-FIXES-NEXT:   ~FooBar3();
// CHECK-FIXES-NEXT: };
class FooBar3 {
protected:
  CONCAT(vir, tual) ~FooBar3();
};

// CHECK-MESSAGES: :[[@LINE+6]]:7: warning: destructor of 'FooBar4' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+5]]:7: note: make it protected and non-virtual
// CHECK-FIXES:      class FooBar4 {
// CHECK-FIXES-NEXT: protected:
// CHECK-FIXES-NEXT:   ~CONCAT(Foo, Bar4());
// CHECK-FIXES-NEXT: };
class FooBar4 {
protected:
  CONCAT(vir, tual) ~CONCAT(Foo, Bar4());
};

// CHECK-MESSAGES: :[[@LINE+3]]:7: warning: destructor of 'FooBar5' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+2]]:7: note: make it protected and non-virtual
#define XMACRO(COLUMN1, COLUMN2) COLUMN1 COLUMN2
class FooBar5 {
protected:
  XMACRO(CONCAT(vir, tual), ~CONCAT(Foo, Bar5());) // no-crash, no-fixit
};
#undef XMACRO
#undef CONCAT
} // namespace macro_tests

namespace FinalClassCannotBeBaseClass {
class Base {
public:
  Base() = default;
  virtual void func() = 0;

protected:
  ~Base() = default;
};

// no-warning: 'MostDerived' cannot be a base class, since it's marked 'final'.
class MostDerived final : public Base {
public:
  MostDerived() = default;
  ~MostDerived() = default;
  void func() final;
};
} // namespace FinalClassCannotBeBaseClass

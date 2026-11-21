// RUN: %check_clang_tidy %s readability-redundant-override %t
// RUN: %clang_cc1 -std=c++11 -fsyntax-only %t.cpp
// RUN: %check_clang_tidy %s readability-redundant-override %t-linux -- -- -target x86_64-unknown-linux-gnu
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -fsyntax-only %t-linux.cpp

struct Base {
  virtual void f() {}
};

struct Inline : Base {
  void f() override { Base::f(); }
};
// CHECK-MESSAGES: :[[@LINE-2]]:8: warning: redundant override
// CHECK-FIXES: struct Inline : Base {
// CHECK-FIXES-NOT: f()
// CHECK-FIXES: };

struct VoidReturn : Base {
  void f() override { return Base::f(); }
};
// CHECK-MESSAGES: :[[@LINE-2]]:8: warning: redundant override
// CHECK-FIXES: struct VoidReturn : Base {
// CHECK-FIXES-NOT: f()
// CHECK-FIXES: };

struct ValueBase {
  virtual int f(int value, const int &other) const { return value + other; }
};
struct Value : ValueBase {
  int f(int value, const int &other) const override {
    return ValueBase::f(value, other);
  }
};
// CHECK-MESSAGES: :[[@LINE-4]]:7: warning: redundant override
// CHECK-FIXES: struct Value : ValueBase {
// CHECK-FIXES-NOT: f(
// CHECK-FIXES: };

struct RefBase {
  virtual int &f(int &value) { return value; }
};
struct Reference : RefBase {
  int &f(int &value) override { return RefBase::f(value); }
};
// CHECK-MESSAGES: :[[@LINE-2]]:8: warning: redundant override
// CHECK-FIXES: struct Reference : RefBase {
// CHECK-FIXES-NOT: f(
// CHECK-FIXES: };

struct SourceDefinition : Base {
  void f() override;
};
void SourceDefinition::f() { Base::f(); }
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: redundant override
// CHECK-FIXES: struct SourceDefinition : Base {
// CHECK-FIXES-NOT: f()
// CHECK-FIXES: };
// CHECK-FIXES-NOT: SourceDefinition::f

// Compare access after accounting for inheritance, not just the two methods.
struct SameProtected : protected Base {
protected:
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct SameProtected : protected Base {
// CHECK-FIXES-NEXT: protected:
// CHECK-FIXES-NEXT: void f() override { Base::f(); }
// CHECK-FIXES: };

struct SamePrivate : private Base {
private:
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct SamePrivate : private Base {
// CHECK-FIXES-NEXT: private:
// CHECK-FIXES-NEXT: void f() override { Base::f(); }
// CHECK-FIXES: };

struct Narrowed : Base {
protected:
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct Narrowed : Base {
// CHECK-FIXES: void f() override { Base::f(); }

struct Published : protected Base {
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct Published : protected Base {
// CHECK-FIXES-NEXT: void f() override { Base::f(); }

struct ProtectedBase {
protected:
  virtual void f() {}
};
struct Widened : ProtectedBase {
  void f() override { ProtectedBase::f(); }
};
// CHECK-FIXES: struct Widened : ProtectedBase {
// CHECK-FIXES-NEXT: void f() override { ProtectedBase::f(); }

struct PrivatePublished : private Base {
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct PrivatePublished : private Base {
// CHECK-FIXES-NEXT: void f() override { Base::f(); }

struct Final : Base {
  void f() final { Base::f(); }
};
// CHECK-FIXES: void f() final { Base::f(); }

struct PureBase {
  virtual void f() = 0;
};
void PureBase::f() {}
struct Concrete : PureBase {
  void f() override { PureBase::f(); }
};
// CHECK-FIXES: void f() override { PureBase::f(); }

struct DefaultBase {
  virtual int f(int value = 1) { return value; }
};
struct Defaults : DefaultBase {
  int f(int value = 2) override { return DefaultBase::f(value); }
};
// CHECK-FIXES: int f(int value = 2) override { return DefaultBase::f(value); }

struct DifferentNoexcept : Base {
  void f() noexcept override { Base::f(); }
};
// CHECK-FIXES: void f() noexcept override { Base::f(); }

struct Attributed : Base {
  __attribute__((noinline)) void f() override { Base::f(); }
};
// CHECK-FIXES: __attribute__((noinline)) void f() override { Base::f(); }

struct OverloadedBase {
  virtual void f() {}
  void f(int) {}
};
struct HidesOverload : OverloadedBase {
  void f() override { OverloadedBase::f(); }
};
// CHECK-FIXES: void f() override { OverloadedBase::f(); }

struct OwnOverload : Base {
  void f() override { Base::f(); }
  void f(int) {}
};
// CHECK-FIXES: void f() override { Base::f(); }

struct Observed : Base {
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct Observed : Base {
// CHECK-FIXES-NEXT: void f() override { Base::f(); }
auto MemberPointer = &Observed::f;

struct Copyable {};
struct CopyBase {
  virtual void f(Copyable) {}
};
struct ExtraCopy : CopyBase {
  void f(Copyable value) override { CopyBase::f(value); }
};
// CHECK-FIXES: void f(Copyable value) override { CopyBase::f(value); }

struct ResultBase {
  virtual Copyable f() { return {}; }
};
struct ExtraResult : ResultBase {
  Copyable f() override { return ResultBase::f(); }
};
// CHECK-FIXES: Copyable f() override { return ResultBase::f(); }

struct ExtraWork : Base {
  void f() override {
    Base::f();
    Base::f();
  }
};
// CHECK-FIXES: struct ExtraWork : Base {
// CHECK-FIXES-NEXT: void f() override {
// CHECK-FIXES-NEXT: Base::f();
// CHECK-FIXES-NEXT: Base::f();

struct ChangedArgument : ValueBase {
  int f(int value, const int &other) const override {
    return ValueBase::f(value + 1, other);
  }
};
// CHECK-FIXES: return ValueBase::f(value + 1, other);

struct EmptyBase {};
struct MultipleBases : Base, EmptyBase {
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct MultipleBases : Base, EmptyBase {
// CHECK-FIXES-NEXT: void f() override { Base::f(); }

struct VirtualBase : virtual Base {
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct VirtualBase : virtual Base {
// CHECK-FIXES-NEXT: void f() override { Base::f(); }

template <typename T> struct Template : Base {
  void f() override { Base::f(); }
};
// CHECK-FIXES: void f() override { Base::f(); }
Template<int> Instantiated;

// Shared declarations must not result in only the definition being removed.
struct Shared : Base {
  void f() override, g();
};
void Shared::f() { Base::f(); }
// CHECK-FIXES: void f() override, g();
// CHECK-FIXES: void Shared::f() { Base::f(); }

struct PrivateNarrowed : Base {
private:
  void f() override { Base::f(); }
};
// CHECK-FIXES: struct PrivateNarrowed : Base {
// CHECK-FIXES: void f() override { Base::f(); }

struct SameProtectedMember : ProtectedBase {
protected:
  void f() override { ProtectedBase::f(); }
};
// CHECK-FIXES: struct SameProtectedMember : ProtectedBase {
// CHECK-FIXES-NEXT: protected:
// CHECK-FIXES-NEXT: void f() override { ProtectedBase::f(); }
// CHECK-FIXES: };

struct PrivateBase {
  friend struct PrivateForward;

private:
  virtual void f() {}
};
struct PrivateForward : PrivateBase {
private:
  void f() override { PrivateBase::f(); }
};
// CHECK-FIXES: void f() override { PrivateBase::f(); }

struct OtherObject : Base {
  Base other;
  void f() override { other.Base::f(); }
};
// CHECK-FIXES: void f() override { other.Base::f(); }

struct ObservedBefore : Base {
  void f() override;
};
auto EarlierPointer = &ObservedBefore::f;
void ObservedBefore::f() { Base::f(); }
// CHECK-FIXES: auto EarlierPointer = &ObservedBefore::f;
// CHECK-FIXES-NEXT: void ObservedBefore::f() { Base::f(); }

struct IndirectBase {
  virtual void f() {}
};
struct Indirect : IndirectBase {
  void (IndirectBase::*target)() = &IndirectBase::f;
  void f() override { (this->*target)(); }
};
// CHECK-FIXES: void f() override { (this->*target)(); }

#define FORWARD Base::f()
struct MacroBody : Base {
  void f() override { FORWARD; }
};
// CHECK-FIXES: void f() override { FORWARD; }

#define METHOD_DECL void f() override;
struct MacroDeclaration : Base {
  METHOD_DECL
};
void MacroDeclaration::f() { Base::f(); }
// CHECK-FIXES: METHOD_DECL
// CHECK-FIXES: void MacroDeclaration::f() { Base::f(); }

struct ConditionalBody : Base {
  void f() override {
#if 1
    Base::f();
#else
    Base::f();
    Base::f();
#endif
  }
};
// CHECK-FIXES: struct ConditionalBody : Base {
// CHECK-FIXES-NEXT: void f() override {
// CHECK-FIXES-NEXT: #if 1
// CHECK-FIXES-NEXT: Base::f();

struct InheritedDefaults : DefaultBase {
  int f(int value) override { return DefaultBase::f(value); }
};
// CHECK-FIXES: int f(int value) override { return DefaultBase::f(value); }

struct CovariantBase {
  virtual CovariantBase *f() { return this; }
};
struct Covariant : CovariantBase {
  Covariant *f() override { return static_cast<Covariant *>(CovariantBase::f()); }
};
// CHECK-FIXES: Covariant *f() override { return static_cast<Covariant *>(CovariantBase::f()); }

struct ParameterBase {
  virtual void f(int) {}
};
struct AnnotatedParameter : ParameterBase {
  void f(int value __attribute__((annotate("parameter")))) override {
    ParameterBase::f(value);
  }
};
// CHECK-FIXES: void f(int value __attribute__((annotate("parameter")))) override {
// CHECK-FIXES-NEXT: ParameterBase::f(value);

// Equal protected access does not preserve access from the base class itself.
struct AccessBase {
protected:
  virtual void f() {}

public:
  template <typename T> static auto accessible(T &object)
      -> decltype(object.f(), char());
  static int accessible(...);
};
struct AccessDerived : AccessBase {
protected:
  void f() override { AccessBase::f(); }
};
// CHECK-FIXES: void f() override { AccessBase::f(); }
using AccessResult =
    decltype(AccessBase::accessible(*static_cast<AccessDerived *>(nullptr)));
static_assert(sizeof(AccessResult) == sizeof(int), "preserve protected access");

// A semicolon after a definition is a separate empty declaration.
// clang-format off
struct Trailing : Base {
  void f() override { Base::f(); }
  // Keep the comment with the following empty declaration.
  ;
};
// clang-format on
// CHECK-MESSAGES: :[[@LINE-5]]:8: warning: redundant override
// CHECK-FIXES: struct Trailing : Base {
// CHECK-FIXES-NOT: f()
// CHECK-FIXES: // Keep the comment with the following empty declaration.
// CHECK-FIXES-NEXT: ;
// CHECK-FIXES-NEXT: };

// Ordinary calls continue to resolve to the inherited method after fixing.
void use() {
  Inline().f();
  VoidReturn().f();
  SourceDefinition().f();
  int value = 0;
  Value().f(value, value);
  Reference().f(value);
}

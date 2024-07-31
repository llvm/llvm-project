// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-unused-local-non-trivial-variable %t -- \
// RUN:       -config="{CheckOptions: {bugprone-unused-local-non-trivial-variable.IncludeTypes: '::async::Future;::async::Foo.*', bugprone-unused-local-non-trivial-variable.ExcludeTypes: '::async::FooBar'}}" \
// RUN:       -- -fexceptions

namespace async {
template <typename T>
class Ptr {
  public:
  explicit Ptr(T Arg) : Underlying(new T(Arg)) {}
  T& operator->() {
    return Underlying;
  }
  ~Ptr() {
    delete Underlying;
  }
  private:
    T* Underlying;
};

template<typename T>
class Future {
public:
    T get() {
        return Pending;
    }
    ~Future();
private:
    T Pending;
};

class FooBar {
  public:
    ~FooBar();
  private:
    Future<int> Fut;
};

class FooQux {
  public:
    ~FooQux();
  private:
    Future<int> Fut;
};

class FizzFoo {
  public:
    ~FizzFoo();
  private:
    Future<int> Fut;
};

} // namespace async

// Warning is still emitted if there are type aliases.
namespace a {
template<typename T>
using Future = async::Future<T>;
} // namespace a

void releaseUnits();
struct Units {
  ~Units() {
    releaseUnits();
  }
};
a::Future<Units> acquireUnits();

template<typename T>
T qux(T Generic) {
    async::Future<Units> PendingA = acquireUnits();
    auto PendingB = acquireUnits();
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unused local variable 'PendingB' of type 'a::Future<Units>' (aka 'Future<Units>') [bugprone-unused-local-non-trivial-variable]
    async::Future<Units> MustBeUsed;
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: unused local variable 'MustBeUsed' of type 'async::Future<Units>' [bugprone-unused-local-non-trivial-variable]
    PendingA.get();
    async::Future<T> TemplateType;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: unused local variable 'TemplateType' of type 'async::Future<T>' [bugprone-unused-local-non-trivial-variable]
    a::Future<T> AliasTemplateType;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: unused local variable 'AliasTemplateType' of type 'a::Future<T>' (aka 'Future<type-parameter-0-0>') [bugprone-unused-local-non-trivial-variable]
    [[maybe_unused]] async::Future<Units> MaybeUnused;
    return Generic;
}

async::Future<int> Global;

int bar(int Num) {
    a::Future<Units> PendingA = acquireUnits();
    a::Future<Units> PendingB = acquireUnits(); // not used at all, unused variable not fired because of destructor side effect
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: unused local variable 'PendingB' of type 'a::Future<Units>' (aka 'Future<Units>') [bugprone-unused-local-non-trivial-variable]
    auto Num2 = PendingA.get();
    auto Num3 = qux(Num);
    async::Ptr<a::Future<Units>> Shared = async::Ptr<a::Future<Units>>(acquireUnits());
    static auto UnusedStatic = async::Future<Units>();
    thread_local async::Future<Units> UnusedThreadLocal;
    auto Captured = acquireUnits();
    Num3 += [Captured]() {
      return 1;
    }();
    a::Future<Units> Referenced = acquireUnits();
    a::Future<Units>* Pointer = &Referenced;
    a::Future<Units>& Reference = Referenced;
    const a::Future<Units>& ConstReference = Referenced;
    try {
    } catch (a::Future<Units> Fut) {
    }
    struct Holder {
      a::Future<Units> Fut;
    };
    Holder H;
    auto [fut] = H;
    return Num * Num3;
}

void exclusion() {
  async::FizzFoo A;
  async::FooBar B;
  async::FooQux C;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: unused local variable 'C' of type 'async::FooQux' [bugprone-unused-local-non-trivial-variable]
}

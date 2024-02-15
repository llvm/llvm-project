// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

void WTFBreakpointTrap();
void WTFCrashWithInfo(int, const char*, const char*, int);

inline void compilerFenceForCrash()
{
    asm volatile("" ::: "memory");
}

inline void isIntegralOrPointerType() { }

template<typename T, typename... Types>
void isIntegralOrPointerType(T, Types... types)
{
    static_assert(sizeof(char) < sizeof(short), "All types need to be bitwise_cast-able to integral type for logging");
    isIntegralOrPointerType(types...);
}

#define CRASH_WITH_INFO(...) do { \
    isIntegralOrPointerType(__VA_ARGS__); \
    compilerFenceForCrash(); \
    WTFBreakpointTrap(); \
    __builtin_unreachable(); \
} while (0)

#define RELEASE_ASSERT(assertion, ...) do { \
    if (!(assertion)) \
        CRASH_WITH_INFO(__VA_ARGS__); \
} while (0)

#if !defined(NOT_TAIL_CALLED)
#if __has_attribute(not_tail_called)
#define NOT_TAIL_CALLED __attribute__((not_tail_called))
#else
#define NOT_TAIL_CALLED
#endif
#endif
#define NO_RETURN_DUE_TO_CRASH

#if !defined(ALWAYS_INLINE)
#define ALWAYS_INLINE inline
#endif

NO_RETURN_DUE_TO_CRASH NOT_TAIL_CALLED void WTFCrashWithInfoImpl(int line, const char* file, const char* function, int counter, unsigned long reason);
NO_RETURN_DUE_TO_CRASH NOT_TAIL_CALLED void WTFCrashWithInfo(int line, const char* file, const char* function, int counter);

template<typename T>
ALWAYS_INLINE unsigned long wtfCrashArg(T* arg) { return reinterpret_cast<unsigned long>(arg); }

template<typename T>
ALWAYS_INLINE unsigned long wtfCrashArg(T arg) { return arg; }

template<typename T>
NO_RETURN_DUE_TO_CRASH ALWAYS_INLINE void WTFCrashWithInfo(int line, const char* file, const char* function, int counter, T reason)
{
    WTFCrashWithInfoImpl(line, file, function, counter, wtfCrashArg(reason));
}

enum class Flags : unsigned short {
  Flag1 = 1 << 0,
  Flag2 = 1 << 1,
  Flag3 = 1 << 2,
};

template<typename E> class OptionSet {
public:
  using StorageType = unsigned short;

  static constexpr OptionSet fromRaw(StorageType rawValue) {
    return OptionSet(static_cast<E>(rawValue), FromRawValue);
  }

  constexpr OptionSet() = default;

  constexpr OptionSet(E e)
    : m_storage(static_cast<StorageType>(e)) {
  }

  constexpr StorageType toRaw() const { return m_storage; }

  constexpr bool isEmpty() const { return !m_storage; }

  constexpr explicit operator bool() const { return !isEmpty(); }

  constexpr bool contains(E option) const { return containsAny(option); }
  constexpr bool containsAny(OptionSet optionSet) const {
    return !!(*this & optionSet);
  }

  constexpr bool containsAll(OptionSet optionSet) const {
    return (*this & optionSet) == optionSet;
  }

  constexpr void add(OptionSet optionSet) { m_storage |= optionSet.m_storage; }

  constexpr void remove(OptionSet optionSet)
  {
      m_storage &= ~optionSet.m_storage;
  }

  constexpr void set(OptionSet optionSet, bool value)
  {
    if (value)
      add(optionSet);
    else
      remove(optionSet);
  }

  constexpr friend OptionSet operator|(OptionSet lhs, OptionSet rhs) {
    return fromRaw(lhs.m_storage | rhs.m_storage);
  }

  constexpr friend OptionSet operator&(OptionSet lhs, OptionSet rhs) {
    return fromRaw(lhs.m_storage & rhs.m_storage);
  }

  constexpr friend OptionSet operator-(OptionSet lhs, OptionSet rhs) {
    return fromRaw(lhs.m_storage & ~rhs.m_storage);
  }

  constexpr friend OptionSet operator^(OptionSet lhs, OptionSet rhs) {
    return fromRaw(lhs.m_storage ^ rhs.m_storage);
  }

private:
  enum InitializationTag { FromRawValue };
  constexpr OptionSet(E e, InitializationTag)
    : m_storage(static_cast<StorageType>(e)) {
  }
  StorageType m_storage { 0 };
};

class Number {
public:
  Number(int v) : v(v) { }
  Number(double);
  Number operator+(const Number&);
  const int& value() const { return v; }
private:
  int v;
};

class RefCounted {
public:
  void ref() const;
  void deref() const;

  void someFunction();
  int otherFunction();

  int trivial1() { return 123; }
  float trivial2() { return 0.3; }
  float trivial3() { return (float)0.4; }
  float trivial4() { return 0.5f; }
  char trivial5() { return 'a'; }
  const char *trivial6() { return "abc"; }
  int trivial7() { return (1); }
  Number trivial8() { return Number { 5 }; }
  int trivial9() { return 3 + 4; }
  int trivial10() { return 0x1010 | 0x1; }
  int trivial11(int v) { return v + 1; }
  const char *trivial12(char *p) { return p ? "str" : "null"; }
  int trivial13(int v) {
    if (v)
      return 123;
    else
      return 0;
  }
  int trivial14(int v) {
    switch (v) {
      case 1:
        return 100;
      case 2:
        return 200;
      default:
        return 300;
    }
    return 0;
  }
  void *trivial15() { return static_cast<void*>(this); }
  unsigned long trivial16() { return reinterpret_cast<unsigned long>(this); }
  RefCounted& trivial17() const { return const_cast<RefCounted&>(*this); }
  RefCounted& trivial18() const { RELEASE_ASSERT(this, "this must be not null"); return const_cast<RefCounted&>(*this); }
  void trivial19() const { return; }

  static constexpr unsigned numBits = 4;
  int trivial20() { return v >> numBits; }

  const int* trivial21() { return number ? &number->value() : nullptr; }

  enum class Enum : unsigned short  {
      Value1 = 1,
      Value2 = 2,
  };
  bool trivial22() { return enumValue == Enum::Value1; }

  bool trivial23() const { return OptionSet<Flags>::fromRaw(v).contains(Flags::Flag1); }

  static RefCounted& singleton() {
    static RefCounted s_RefCounted;
    s_RefCounted.ref();
    return s_RefCounted;
  }

  Number nonTrivial1() { return Number(3) + Number(4); }
  Number nonTrivial2() { return Number { 0.3 }; }
  int nonTrivial3() { return v ? otherFunction() : 0; }
  int nonTrivial4() {
    if (v)
      return 8;
    else
      return otherFunction();
  }

  int nonTrivial5() {
    if (v)
      return otherFunction();
    else
      return 9;
  }

  int nonTrivial6() {
    if (otherFunction())
      return 1;
    else
      return 0;
  }

  int nonTrivial7() {
    switch (v) {
      case 1:
        return otherFunction();
      default:
        return 7;
    }
  }

  int nonTrivial8() {
    switch (v) {
      case 1:
        return 9;
      default:
        return otherFunction();
    }
  }

  int nonTrivial9() {
    switch (otherFunction()) {
      case 0:
        return -1;
      default:
        return 12;
    }
  }

  unsigned v { 0 };
  Number* number { nullptr };
  Enum enumValue { Enum::Value1 };
};

RefCounted* refCountedObj();

void test()
{
  refCountedObj()->someFunction();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}

class UnrelatedClass {
  RefPtr<RefCounted> Field;
  bool value;

public:
  RefCounted &getFieldTrivial() { return *Field.get(); }
  RefCounted *getFieldTernary() { return value ? Field.get() : nullptr; }

  void test() {
    getFieldTrivial().trivial1(); // no-warning
    getFieldTrivial().trivial2(); // no-warning
    getFieldTrivial().trivial3(); // no-warning
    getFieldTrivial().trivial4(); // no-warning
    getFieldTrivial().trivial5(); // no-warning
    getFieldTrivial().trivial6(); // no-warning
    getFieldTrivial().trivial7(); // no-warning
    getFieldTrivial().trivial8(); // no-warning
    getFieldTrivial().trivial9(); // no-warning
    getFieldTrivial().trivial10(); // no-warning
    getFieldTrivial().trivial11(1); // no-warning
    getFieldTrivial().trivial12(nullptr); // no-warning
    getFieldTrivial().trivial13(0); // no-warning
    getFieldTrivial().trivial14(3); // no-warning
    getFieldTrivial().trivial15(); // no-warning
    getFieldTrivial().trivial16(); // no-warning
    getFieldTrivial().trivial17(); // no-warning
    getFieldTrivial().trivial18(); // no-warning
    getFieldTrivial().trivial19(); // no-warning
    getFieldTrivial().trivial20(); // no-warning
    getFieldTrivial().trivial21(); // no-warning
    getFieldTrivial().trivial22(); // no-warning
    getFieldTrivial().trivial23(); // no-warning
    RefCounted::singleton().trivial18(); // no-warning
    RefCounted::singleton().someFunction(); // no-warning

    getFieldTrivial().someFunction();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial1();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial2();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial3();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial4();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial5();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial6();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial7();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial8();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial9();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
  }
};

class UnrelatedClass2 {
  RefPtr<UnrelatedClass> Field;

public:
  UnrelatedClass &getFieldTrivial() { return *Field.get(); }
  RefCounted &getFieldTrivialRecursively() { return getFieldTrivial().getFieldTrivial(); }
  RefCounted *getFieldTrivialTernary() { return Field ? Field->getFieldTernary() : nullptr; }

  void test() {
    getFieldTrivialRecursively().trivial1(); // no-warning
    getFieldTrivialTernary()->trivial2(); // no-warning
    getFieldTrivialRecursively().someFunction();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
  }
};

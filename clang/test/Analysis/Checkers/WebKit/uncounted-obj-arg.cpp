// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

void WTFBreakpointTrap();
void WTFCrashWithInfo(int, const char*, const char*, int);
void WTFReportAssertionFailure(const char* file, int line, const char* function, const char* assertion);
void WTFReportBacktrace(void);

void WTFCrash(void);
void WTFCrashWithSecurityImplication(void);

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

#define ASSERT(assertion, ...) do { \
    if (!(assertion)) { \
        WTFReportAssertionFailure(__FILE__, __LINE__, __PRETTY_FUNCTION__, #assertion); \
        CRASH_WITH_INFO(__VA_ARGS__); \
    } \
} while (0)

#if !defined(ALWAYS_INLINE)
#define ALWAYS_INLINE inline
#endif

void WTFCrashWithInfoImpl(int line, const char* file, const char* function, int counter, unsigned long reason);
void WTFCrashWithInfo(int line, const char* file, const char* function, int counter);

template<typename T>
ALWAYS_INLINE unsigned long wtfCrashArg(T* arg) { return reinterpret_cast<unsigned long>(arg); }

template<typename T>
ALWAYS_INLINE unsigned long wtfCrashArg(T arg) { return arg; }

template<typename T>
void WTFCrashWithInfo(int line, const char* file, const char* function, int counter, T reason)
{
    WTFCrashWithInfoImpl(line, file, function, counter, wtfCrashArg(reason));
}

template<typename ToType, typename FromType>
ToType bitwise_cast(FromType from);

namespace std {

template<typename T>
T* addressof(T& arg);

template<typename T>
T&& forward(T& arg);

template<typename T>
T&& move( T&& t );

} // namespace std

bool isMainThread();
bool isMainThreadOrGCThread();
bool isMainRunLoop();
bool isWebThread();
bool isUIThread();
bool mayBeGCThread();

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

int atoi(const char* str);

class Number {
public:
  Number(int v) : v(v) { }
  Number(double);
  Number(const char* str) : v(atoi(str)) { }
  Number operator+(const Number&);
  Number& operator++() { ++v; return *this; }
  Number operator++(int) { Number returnValue(v); ++v; return returnValue; }
  const int& value() const { return v; }
  void someMethod();

private:
  int v;
};

class DerivedNumber : public Number {
public:
  DerivedNumber(char c) : Number(c - '0') { }
  DerivedNumber(const char* str) : Number(atoi(str)) { }
};

class ComplexNumber {
public:
  ComplexNumber() : realPart(0), complexPart(0) { }
  ComplexNumber(int real, const char* str) : realPart(real), complexPart(str) { }
  ComplexNumber(const ComplexNumber&);
  ComplexNumber& operator++() { realPart.someMethod(); return *this; }
  ComplexNumber operator++(int);
  ComplexNumber& operator<<(int);
  ComplexNumber& operator+();

  const Number& real() const { return realPart; }

private:
  Number realPart;
  Number complexPart;
};

class ObjectWithNonTrivialDestructor {
public:
  ObjectWithNonTrivialDestructor() { }
  ObjectWithNonTrivialDestructor(unsigned v) : v(v) { }
  ~ObjectWithNonTrivialDestructor() { }

  unsigned value() const { return v; }

private:
  unsigned v { 0 };
};

class ObjectWithMutatingDestructor {
public:
  ObjectWithMutatingDestructor() : n(0) { }
  ObjectWithMutatingDestructor(int n) : n(n) { }
  ~ObjectWithMutatingDestructor() { n.someMethod(); }

  unsigned value() const { return n.value(); }

private:
  Number n;
};

class RefCounted {
public:
  void ref() const;
  void deref() const;

  void method();
  void someFunction();
  int otherFunction();
  unsigned recursiveTrivialFunction(int n) { return !n ? 1 : recursiveTrivialFunction(n - 1);  }
  unsigned recursiveComplexFunction(int n) { return !n ? otherFunction() : recursiveComplexFunction(n - 1);  }
  unsigned mutuallyRecursiveFunction1(int n) { return n < 0 ? 1 : (n % 2 ? mutuallyRecursiveFunction2(n - 2) : mutuallyRecursiveFunction1(n - 1)); }
  unsigned mutuallyRecursiveFunction2(int n) { return n < 0 ? 1 : (n % 3 ? mutuallyRecursiveFunction2(n - 3) : mutuallyRecursiveFunction1(n - 2)); }
  unsigned mutuallyRecursiveFunction3(int n) { return n < 0 ? 1 : (n % 5 ? mutuallyRecursiveFunction3(n - 5) : mutuallyRecursiveFunction4(n - 3)); }
  unsigned mutuallyRecursiveFunction4(int n) { return n < 0 ? 1 : (n % 7 ? otherFunction() : mutuallyRecursiveFunction3(n - 3)); }
  unsigned recursiveFunction5(unsigned n) { return n > 100 ? 2 : (n % 2 ? recursiveFunction5(n + 1) : recursiveFunction6(n + 2)); }
  unsigned recursiveFunction6(unsigned n) { return n > 100 ? 3 : (n % 2 ? recursiveFunction6(n % 7) : recursiveFunction7(n % 5)); }
  unsigned recursiveFunction7(unsigned n) { return n > 100 ? 5 : recursiveFunction7(n * 5); }

  void mutuallyRecursive8() { mutuallyRecursive9(); someFunction(); }
  void mutuallyRecursive9() { mutuallyRecursive8(); }

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
  unsigned long trivial16() { return *reinterpret_cast<unsigned long*>(this); }
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
  int trivial24() const { ASSERT(v); return v; }
  unsigned trivial25() const { return __c11_atomic_load((volatile _Atomic(unsigned) *)&v, __ATOMIC_RELAXED); }
  bool trivial26() { bool hasValue = v; return !hasValue; }
  bool trivial27(int v) { bool value; value = v ? 1 : 0; return value; }
  bool trivial28() { return true; }
  bool trivial29() { return false; }
  unsigned trivial30(unsigned v) { unsigned r = 0xff; r |= v; return r; }
  int trivial31(int* v) { return v[0]; }
  unsigned trivial32() { return sizeof(int); }
  unsigned trivial33() { return ~0xff; }
  template <unsigned v> unsigned trivial34() { return v; }
  void trivial35() { v++; }
  void trivial36() { ++(*number); }
  void trivial37() { (*number)++; }
  void trivial38() { v++; if (__builtin_expect(!!(number), 1)) (*number)++; }
  int trivial39() { return -v; }
  int trivial40() { return v << 2; }
  unsigned trivial41() { v = ++s_v; return v; }
  unsigned trivial42() { return bitwise_cast<unsigned long>(nullptr); }
  Number* trivial43() { return std::addressof(*number); }
  Number* trivial44() { return new Number(1); }
  ComplexNumber* trivial45() { return new ComplexNumber(); }
  void trivial46() { ASSERT(isMainThread()); }
  void trivial47() { ASSERT(isMainThreadOrGCThread()); }
  void trivial48() { ASSERT(isMainRunLoop()); }
  void trivial49() { ASSERT(isWebThread()); }
  void trivial50() { ASSERT(isUIThread()); }
  void trivial51() { ASSERT(mayBeGCThread()); }
  void trivial52() { WTFCrash(); }
  void trivial53() { WTFCrashWithSecurityImplication(); }
  unsigned trivial54() { return ComplexNumber().real().value(); }
  Number&& trivial55() { return std::forward(*number); }
  unsigned trivial56() { Number n { 5 }; return std::move(n).value(); }
  void trivial57() { do { break; } while (1); }
  void trivial58() { do { continue; } while (0); }
  void trivial59() {
    do { goto label; }
    while (0);
  label:
    return;
  }
  unsigned trivial60() { return ObjectWithNonTrivialDestructor { 5 }.value(); }
  unsigned trivial61() { return DerivedNumber('7').value(); }
  void trivial62() { WTFReportBacktrace(); }

  static RefCounted& singleton() {
    static RefCounted s_RefCounted;
    s_RefCounted.ref();
    return s_RefCounted;
  }

  static RefCounted& otherSingleton() {
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

  static unsigned* another();
  unsigned nonTrivial10() const {
    return __c11_atomic_load((volatile _Atomic(unsigned) *)another(), __ATOMIC_RELAXED);
  }

  void nonTrivial11() {
    Number num(0.3);
  }

  bool nonTrivial12() {
    bool val = otherFunction();
    return val;
  }

  int nonTrivial13() { return ~otherFunction(); }
  int nonTrivial14() { int r = 0xff; r |= otherFunction(); return r; }
  void nonTrivial15() { ++complex; }
  void nonTrivial16() { complex++; }
  ComplexNumber nonTrivial17() { return complex << 2; }
  ComplexNumber nonTrivial18() { return +complex; }
  ComplexNumber* nonTrivial19() { return new ComplexNumber(complex); }
  unsigned nonTrivial20() { return ObjectWithMutatingDestructor { 7 }.value(); }
  unsigned nonTrivial21() { return Number("123").value(); }
  unsigned nonTrivial22() { return ComplexNumber(123, "456").real().value(); }
  unsigned nonTrivial23() { return DerivedNumber("123").value(); }

  static unsigned s_v;
  unsigned v { 0 };
  Number* number { nullptr };
  ComplexNumber complex;
  Enum enumValue { Enum::Value1 };
};

unsigned RefCounted::s_v = 0;

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
    getFieldTrivial().trivial24(); // no-warning
    getFieldTrivial().trivial25(); // no-warning
    getFieldTrivial().trivial26(); // no-warning
    getFieldTrivial().trivial27(5); // no-warning
    getFieldTrivial().trivial28(); // no-warning
    getFieldTrivial().trivial29(); // no-warning
    getFieldTrivial().trivial30(7); // no-warning
    int a[] = {1, 2};
    getFieldTrivial().trivial31(a); // no-warning
    getFieldTrivial().trivial32(); // no-warning
    getFieldTrivial().trivial33(); // no-warning
    getFieldTrivial().trivial34<7>(); // no-warning
    getFieldTrivial().trivial35(); // no-warning
    getFieldTrivial().trivial36(); // no-warning
    getFieldTrivial().trivial37(); // no-warning
    getFieldTrivial().trivial38(); // no-warning
    getFieldTrivial().trivial39(); // no-warning
    getFieldTrivial().trivial40(); // no-warning
    getFieldTrivial().trivial41(); // no-warning
    getFieldTrivial().trivial42(); // no-warning
    getFieldTrivial().trivial43(); // no-warning
    getFieldTrivial().trivial44(); // no-warning
    getFieldTrivial().trivial45(); // no-warning
    getFieldTrivial().trivial46(); // no-warning
    getFieldTrivial().trivial47(); // no-warning
    getFieldTrivial().trivial48(); // no-warning
    getFieldTrivial().trivial49(); // no-warning
    getFieldTrivial().trivial50(); // no-warning
    getFieldTrivial().trivial51(); // no-warning
    getFieldTrivial().trivial52(); // no-warning
    getFieldTrivial().trivial53(); // no-warning
    getFieldTrivial().trivial54(); // no-warning
    getFieldTrivial().trivial55(); // no-warning
    getFieldTrivial().trivial56(); // no-warning
    getFieldTrivial().trivial57(); // no-warning
    getFieldTrivial().trivial58(); // no-warning
    getFieldTrivial().trivial59(); // no-warning
    getFieldTrivial().trivial60(); // no-warning
    getFieldTrivial().trivial61(); // no-warning
    getFieldTrivial().trivial62(); // no-warning

    RefCounted::singleton().trivial18(); // no-warning
    RefCounted::singleton().someFunction(); // no-warning
    RefCounted::otherSingleton().trivial18(); // no-warning
    RefCounted::otherSingleton().someFunction(); // no-warning

    getFieldTrivial().recursiveTrivialFunction(7); // no-warning
    getFieldTrivial().recursiveComplexFunction(9);
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().mutuallyRecursiveFunction1(11); // no-warning
    getFieldTrivial().mutuallyRecursiveFunction2(13); // no-warning
    getFieldTrivial().mutuallyRecursiveFunction3(17);
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().mutuallyRecursiveFunction4(19);
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().recursiveFunction5(23); // no-warning
    getFieldTrivial().recursiveFunction6(29); // no-warning
    getFieldTrivial().recursiveFunction7(31); // no-warning

    getFieldTrivial().mutuallyRecursive8();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().mutuallyRecursive9();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}

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
    getFieldTrivial().nonTrivial10();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial11();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial12();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial13();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial14();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial15();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial16();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial17();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial18();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial19();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial20();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial21();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial22();
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
    getFieldTrivial().nonTrivial23();
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

RefPtr<RefCounted> object();
void someFunction(const RefCounted&);

void test2() {
    someFunction(*object());
}

void system_header() {
  callMethod<RefCountable>(object);
}

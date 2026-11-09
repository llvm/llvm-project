// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.MemoryUnsafeCastChecker -std=c++20 -verify %s
// expected-no-diagnostics

// Explicit casts producing a prvalue of record type construct a new object
// through a converting constructor or a conversion operator (or are a no-op
// copy), so they don't reinterpret memory. The redeclarations below make the
// source and destination types refer to different redeclarations of the same
// class.

class Base { };
class Derived : public Base { };

class FloatRect {
public:
  FloatRect();
};

class LayoutRect {
public:
  LayoutRect();
  LayoutRect(const FloatRect&);
  operator FloatRect() const;
};

FloatRect makeFloatRect();

class Base;
class Derived;
class FloatRect;
class LayoutRect;

void test_value_conversions(const LayoutRect& layoutRect, const FloatRect& floatRect) {
  FloatRect a = FloatRect(layoutRect); // no warning
  FloatRect b = FloatRect(makeFloatRect()); // no warning
  FloatRect c = FloatRect(floatRect); // no warning
  FloatRect d = static_cast<FloatRect>(layoutRect); // no warning
  FloatRect e = static_cast<FloatRect>(makeFloatRect()); // no warning
  FloatRect f = (FloatRect)layoutRect; // no warning
  LayoutRect g = LayoutRect(floatRect); // no warning
  LayoutRect h = static_cast<LayoutRect>(makeFloatRect()); // no warning
}

void test_derived_to_base_value(Derived& derived) {
  Base base = static_cast<Base>(derived); // no warning
}

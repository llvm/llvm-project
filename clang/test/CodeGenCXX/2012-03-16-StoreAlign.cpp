// RUN: %clang_cc1 -emit-llvm -o - -triple x86_64-apple-darwin %s | FileCheck %s

struct Length {
  Length(double v) {
    m_floatValue = static_cast<float>(v);
  }

  bool operator==(const Length& o) const {
    return getFloatValue() == o.getFloatValue();
  }
  bool operator!=(const Length& o) const { return !(*this == o); }
private:
  float getFloatValue() const {
    return m_floatValue;
  }
  float m_floatValue;
};


struct Foo {
  static Length inchLength(double inch);
  static bool getPageSizeFromName(const Length &A) {
    static const Length legalWidth = inchLength(8.5);
    if (A != legalWidth) return true;
    return false;
  }
};

// CHECK: @_ZZN3Foo19getPageSizeFromNameERK6LengthE10legalWidth = linkonce_odr global %struct.Length zeroinitializer, align 4
// CHECK: store float %{{.*}}, ptr @_ZZN3Foo19getPageSizeFromNameERK6LengthE10legalWidth, align 4

bool bar(Length &b) {
  Foo f;
  return f.getPageSizeFromName(b);
}

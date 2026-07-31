// Test for debug info for C++ defaulted member functions

// Supported: -O0, standalone DI
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu %s -o - \
// RUN:   -O0 -disable-llvm-passes \
// RUN:   -debug-info-kind=standalone \
// RUN: | FileCheck %s -check-prefix=ATTR

// ATTR-DAG: DISubprogram(name: "DefaultedOutline", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass
// ATTR-DAG: DISubprogram(name: "DefaultedOutline", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass
// ATTR-DAG: DISubprogram(name: "~DefaultedOutline", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass
// ATTR-DAG: DISubprogram(name: "DefaultedOutline", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass
// ATTR-DAG: DISubprogram(name: "DefaultedOutline", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN16DefaultedOutlineaSERKS_", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass
// ATTR-DAG: DISubprogram(name: "DefaultedOutline", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass
// ATTR-DAG: DISubprogram(name: "DefaultedOutline", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN16DefaultedOutlineaSEOS_", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedOutOfClass

// ATTR-DAG: DISubprogram(name: "DefaultedInline", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "~DefaultedInline", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "DefaultedInline", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN15DefaultedInlineaSERKS_", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "DefaultedInline", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN15DefaultedInlineaSEOS_", {{.*}} spFlags: DISPFlagDefaultedInClass

// ATTR-DAG: DISubprogram(name: "DefaultedInlineWithTemplate", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "~DefaultedInlineWithTemplate", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "DefaultedInlineWithTemplate", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN27DefaultedInlineWithTemplateILi6EEaSERKS0_", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "DefaultedInlineWithTemplate", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN27DefaultedInlineWithTemplateILi6EEaSEOS0_", {{.*}} spFlags: DISPFlagDefaultedInClass

// ATTR-DAG: DISubprogram(name: "DefaultedInlineWithTemplate", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "~DefaultedInlineWithTemplate", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "DefaultedInlineWithTemplate", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN27DefaultedInlineWithTemplateILi7EEaSERKS0_", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "DefaultedInlineWithTemplate", {{.*}} spFlags: DISPFlagDefaultedInClass
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN27DefaultedInlineWithTemplateILi7EEaSEOS0_", {{.*}} spFlags: DISPFlagDefaultedInClass

// ATTR-DAG: DISubprogram(name: "NeverDefaulted", {{.*}} spFlags: DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "~NeverDefaulted", {{.*}} spFlags: DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "NeverDefaulted", {{.*}} spFlags: DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN14NeverDefaultedaSERKS_", {{.*}} spFlags: DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "NeverDefaulted", {{.*}} spFlags: DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "operator=", linkageName: "_ZN14NeverDefaultedaSEOS_", {{.*}} spFlags: DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "NeverDefaulted", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "~NeverDefaulted", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "NeverDefaulted", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedNo
// ATTR-DAG: DISubprogram(name: "~NeverDefaulted", {{.*}} spFlags: DISPFlagDefinition | DISPFlagDefaultedNo

class DefaultedInline {
public:
  DefaultedInline() = default;
  ~DefaultedInline() = default;

  DefaultedInline(const DefaultedInline &) = default;
  DefaultedInline &operator=(const DefaultedInline &) = default;

  DefaultedInline(DefaultedInline &&) = default;
  DefaultedInline &operator=(DefaultedInline &&) = default;
};

class DefaultedOutline {
public:
  DefaultedOutline();
  ~DefaultedOutline();

  DefaultedOutline(const DefaultedOutline &);
  DefaultedOutline &operator=(const DefaultedOutline &);

  DefaultedOutline(DefaultedOutline &&);
  DefaultedOutline &operator=(DefaultedOutline &&);
};

DefaultedOutline::DefaultedOutline() = default;
DefaultedOutline::~DefaultedOutline() = default;

DefaultedOutline::DefaultedOutline(const DefaultedOutline &) = default;
DefaultedOutline &
DefaultedOutline::operator=(const DefaultedOutline &) = default;

DefaultedOutline::DefaultedOutline(DefaultedOutline &&) = default;
DefaultedOutline &DefaultedOutline::operator=(DefaultedOutline &&) = default;

class NeverDefaulted {
public:
  NeverDefaulted() {}
  ~NeverDefaulted() {}

  NeverDefaulted(const NeverDefaulted &) {}
  NeverDefaulted &operator=(const NeverDefaulted &) { return *this; }

  NeverDefaulted(NeverDefaulted &&) {}
  NeverDefaulted &operator=(NeverDefaulted &&) { return *this; }
};

template <int N> class DefaultedInlineWithTemplate {
public:
  char msg[N];

  DefaultedInlineWithTemplate() = default;
  ~DefaultedInlineWithTemplate() = default;

  DefaultedInlineWithTemplate(const DefaultedInlineWithTemplate &) = default;
  DefaultedInlineWithTemplate &operator=(const DefaultedInlineWithTemplate &) = default;

  DefaultedInlineWithTemplate(DefaultedInlineWithTemplate &&) = default;
  DefaultedInlineWithTemplate &operator=(DefaultedInlineWithTemplate &&) = default;
};

int main() {
  DefaultedInline a;
  DefaultedOutline b;
  NeverDefaulted c;
  DefaultedInlineWithTemplate<6> d;
  DefaultedInlineWithTemplate<7> e;
  return 0;
}

// RUN: %check_clang_tidy %s readability-use-explicit-namespaces %t

namespace foo {
void doSomething() {}

template <class T> void doTemplateThing(T &value) { value = value * 2; }

struct StructTest {
  int StructIntMember;
};

class ClassTest {
public:
  int ClassIntMember;
  typedef int ClassTypeDefMember;
};

enum EnumTest {
  EnumValueOne,
  EnumValueTwo,
  EnumValueThree,
};

template <class T> class TemplateClassTest {
public:
  T TemplatizedData;
};

typedef int TypeDefTest;
typedef void (*event_callback)(ClassTest &value);

} // namespace foo

class OutsideNamespace : public foo::ClassTest {
public:
  ClassTypeDefMember UseTypeFromParentOk;
};

OutsideNamespace::ClassTypeDefMember UseTypeFromClassInheritedFromParentOk;

foo::ClassTest AlreadyQualifiedOk() {
  foo::doSomething();
  foo::StructTest first;
  foo::ClassTest second;
  second.ClassIntMember = 55;
  foo::EnumTest picked = foo::EnumValueThree;
  foo::TemplateClassTest<foo::ClassTest> data;
  foo::TemplateClassTest<
      foo::TemplateClassTest<foo::TemplateClassTest<foo::ClassTest>>>
      dataNested;
  foo::StructTest many[8];
  foo::TypeDefTest integer = 22;
  foo::doTemplateThing(integer);
  foo::doTemplateThing<foo::TypeDefTest>(integer);
  struct foo::StructTest fooStruct;
  auto lambdaReturn = []() -> foo::ClassTest { return foo::ClassTest(); };
  auto lambdaTypes = [](foo::StructTest &start,
                        foo::StructTest *end) -> foo::ClassTest {
    return foo::ClassTest();
  };

  foo::ClassTest ConstructOnStack;
  new foo::ClassTest;
  foo::TemplateClassTest<foo::ClassTest> ConstructTemplateOnStack;
  new foo::TemplateClassTest<foo::ClassTest>;
  foo::ClassTest();
  new foo::ClassTest();
  foo::TemplateClassTest<foo::ClassTest>();
  new foo::TemplateClassTest<foo::ClassTest>();
  return foo::ClassTest();
}

namespace foo {
ClassTest InsideNamespaceFooOk() {
  doSomething();
  StructTest first;
  ClassTest second;
  second.ClassIntMember = 55;
  EnumTest picked = EnumValueThree;
  TemplateClassTest<ClassTest> data;
  TemplateClassTest<TemplateClassTest<TemplateClassTest<ClassTest>>> dataNested;
  StructTest many[8];
  TypeDefTest integer = 22;
  doTemplateThing(integer);
  doTemplateThing<TypeDefTest>(integer);
  struct StructTest fooStruct;
  auto lambdaReturn = []() -> ClassTest { return ClassTest(); };
  auto lambdaTypes = [](StructTest &start, StructTest *end) -> ClassTest {
    return ClassTest();
  };

  ClassTest ConstructOnStack;
  new ClassTest;
  TemplateClassTest<ClassTest> ConstructTemplateOnStack;
  new TemplateClassTest<ClassTest>;
  ClassTest();
  new ClassTest();
  TemplateClassTest<ClassTest>();
  new TemplateClassTest<ClassTest>();
  return ClassTest();
}
} // namespace foo

using namespace foo;

ClassTest FixAllMissingFoo()
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: Missing namespace qualifiers
// CHECK-FIXES:  foo::
{
  doSomething();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  StructTest first;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  ClassTest second;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  second.ClassIntMember = 55;
  EnumTest picked = EnumValueThree;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  TemplateClassTest<ClassTest> data;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  TemplateClassTest<TemplateClassTest<TemplateClassTest<ClassTest>>> dataNested;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-5]]:39: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-7]]:57: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  StructTest many[8];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  TypeDefTest integer = 22;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  doTemplateThing(integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  doTemplateThing<TypeDefTest>(integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:19: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  struct StructTest fooStruct;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  auto lambdaReturn = []() -> ClassTest { return ClassTest(); };
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:50: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  auto lambdaTypes = [](StructTest &start, StructTest *end) -> ClassTest {
    return ClassTest();
  };
  // CHECK-MESSAGES: :[[@LINE-3]]:25: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-5]]:44: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-7]]:64: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-8]]:12: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  ClassTest ConstructOnStack;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  new ClassTest;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  TemplateClassTest<ClassTest> ConstructTemplateOnStack;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  new TemplateClassTest<ClassTest>;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:25: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  ClassTest();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  new ClassTest();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  TemplateClassTest<ClassTest>();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  new TemplateClassTest<ClassTest>();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  // CHECK-MESSAGES: :[[@LINE-3]]:25: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
  return ClassTest();
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: Missing namespace qualifiers foo::
  // CHECK-FIXES:  foo::
}

inline namespace TestInlineNamespace {
class InlineNamespaceClassTest {
public:
  int ClassIntMember;
  typedef int ClassTypeDefMember;
};

} // namespace TestInlineNamespace

// inline namespaces should not be qualified because they are used for library
// versioning details
InlineNamespaceClassTest inlineNamespaceOk;
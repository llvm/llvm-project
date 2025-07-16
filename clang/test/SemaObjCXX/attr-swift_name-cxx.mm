// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc -fblocks %s

#define SWIFT_NAME(name) __attribute__((swift_name(name)))
#define SWIFT_ASYNC_NAME(name) __attribute__((__swift_async_name__(name)))

namespace MyNS {
struct NestedStruct {};
}

void nestedStruct_method(MyNS::NestedStruct) SWIFT_NAME("MyNS.NestedStruct.method(self:)");
void nestedStruct_methodConstRef(const MyNS::NestedStruct&) SWIFT_NAME("MyNS.NestedStruct.methodConstRef(self:)");
void nestedStruct_invalidContext1(MyNS::NestedStruct) SWIFT_NAME(".MyNS.NestedStruct.invalidContext1(self:)"); // expected-warning {{'swift_name' attribute has invalid identifier for the context name}}
void nestedStruct_invalidContext2(MyNS::NestedStruct) SWIFT_NAME("MyNS::NestedStruct.invalidContext2(self:)"); // expected-warning {{'swift_name' attribute has invalid identifier for the context name}}
void nestedStruct_invalidContext3(MyNS::NestedStruct) SWIFT_NAME("::MyNS::NestedStruct.invalidContext3(self:)"); // expected-warning {{'swift_name' attribute has invalid identifier for the context name}}
void nestedStruct_invalidContext4(MyNS::NestedStruct) SWIFT_NAME("MyNS..NestedStruct.invalidContext4(self:)"); // expected-warning {{'swift_name' attribute has invalid identifier for the context name}}
void nestedStruct_invalidContext5(MyNS::NestedStruct) SWIFT_NAME("MyNS.NestedStruct.invalidContext5.(self:)"); // expected-warning {{'swift_name' attribute has invalid identifier for the base name}}
void nestedStruct_invalidContext6(MyNS::NestedStruct) SWIFT_NAME("MyNS.NestedStruct::invalidContext6(self:)"); // expected-warning {{'swift_name' attribute has invalid identifier for the base name}}

namespace MyNS {
namespace MyDeepNS {
struct DeepNestedStruct {};
}
}

void deepNestedStruct_method(MyNS::MyDeepNS::DeepNestedStruct) SWIFT_NAME("MyNS.MyDeepNS.DeepNestedStruct.method(self:)");
void deepNestedStruct_methodConstRef(const MyNS::MyDeepNS::DeepNestedStruct&) SWIFT_NAME("MyNS.MyDeepNS.DeepNestedStruct.methodConstRef(self:)");
void deepNestedStruct_invalidContext(const MyNS::MyDeepNS::DeepNestedStruct&) SWIFT_NAME("MyNS::MyDeepNS::DeepNestedStruct.methodConstRef(self:)"); // expected-warning {{'swift_name' attribute has invalid identifier for the context name}}

typedef MyNS::MyDeepNS::DeepNestedStruct DeepNestedStructTypedef;

void deepNestedStructTypedef_method(DeepNestedStructTypedef) SWIFT_NAME("DeepNestedStructTypedef.method(self:)");
void deepNestedStructTypedef_methodQualName(MyNS::MyDeepNS::DeepNestedStruct) SWIFT_NAME("DeepNestedStructTypedef.method(self:)");

struct TopLevelStruct {
  struct StructInStruct {};
};

void structInStruct_method(TopLevelStruct::StructInStruct) SWIFT_NAME("TopLevelStruct.StructInStruct.method(self:)");
void structInStruct_invalidContext(TopLevelStruct::StructInStruct) SWIFT_NAME("TopLevelStruct::StructInStruct.method(self:)"); // expected-warning {{'swift_name' attribute has invalid identifier for the context name}}

typedef int (^CallbackTy)(void);

class CXXClass {
public:
  virtual void doSomethingWithCallback(CallbackTy callback) SWIFT_ASYNC_NAME("doSomething()");

  // expected-warning@+1 {{too few parameters in the signature specified by the '__swift_async_name__' attribute (expected 1; got 0)}}
  virtual void doSomethingWithCallback(int x, CallbackTy callback) SWIFT_ASYNC_NAME("doSomething()");
};

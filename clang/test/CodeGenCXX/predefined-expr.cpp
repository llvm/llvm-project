// RUN: %clang_cc1 -std=c++11 -fblocks %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

// CHECK-DAG: private unnamed_addr constant [15 x i8] c"externFunction\00"
// CHECK-DAG: private unnamed_addr constant [26 x i8] c"void NS::externFunction()\00"
// CHECK-DAG: private unnamed_addr constant [49 x i8] c"void functionTemplateExplicitSpecialization(int)\00"

// CHECK-DAG: private unnamed_addr constant [95 x i8] c"void SpecializedClassTemplate<char>::memberFunctionTemplate(T, U) const [T = char, U = double]\00"
// CHECK-DAG: private unnamed_addr constant [85 x i8] c"void SpecializedClassTemplate<int>::memberFunctionTemplate(int, U) const [U = float]\00"
// CHECK-DAG: private unnamed_addr constant [57 x i8] c"void NonTypeTemplateParam<42>::size() const [Count = 42]\00"
// CHECK-DAG: private unnamed_addr constant [103 x i8] c"static void ClassWithTemplateTemplateParam<char>::staticMember() [T = char, Param = NS::ClassTemplate]\00"
// CHECK-DAG: private unnamed_addr constant [106 x i8] c"void OuterClass<int *>::MiddleClass::InnerClass<float>::memberFunction(T, U) const [T = int *, U = float]\00"
// CHECK-DAG: private unnamed_addr constant [51 x i8] c"void functionTemplateWithCapturedStmt(T) [T = int]\00"
// CHECK-DAG: private unnamed_addr constant [76 x i8] c"auto functionTemplateWithLambda(int)::(anonymous class)::operator()() const\00"
// CHECK-DAG: private unnamed_addr constant [65 x i8] c"void functionTemplateWithUnnamedTemplateParameter(T) [T = float]\00"

// CHECK-DAG: private unnamed_addr constant [60 x i8] c"void functionTemplateExplicitSpecialization(T) [T = double]\00"
// CHECK-DAG: private unnamed_addr constant [52 x i8] c"T *functionTemplateWithCompoundTypes(T *) [T = int]\00" 
// CHECK-DAG: private unnamed_addr constant [54 x i8] c"T functionTemplateWithTemplateReturnType() [T = char]\00"
// CHECK-DAG: private unnamed_addr constant [57 x i8] c"void functionTemplateWithoutParameterList() [T = double]\00"
// CHECK-DAG: private unnamed_addr constant [62 x i8] c"void functionTemplateWithTwoParams(T, U) [T = int, U = float]\00"

// CHECK-DAG: private unnamed_addr constant [22 x i8] c"classTemplateFunction\00"
// CHECK-DAG: private unnamed_addr constant [77 x i8] c"void NS::ClassTemplate<NS::Base *>::classTemplateFunction() [T = NS::Base *]\00"
// CHECK-DAG: private unnamed_addr constant [63 x i8] c"void NS::ClassTemplate<int>::classTemplateFunction() [T = int]\00"

// CHECK-DAG: private unnamed_addr constant [18 x i8] c"functionTemplate1\00"
// CHECK-DAG: private unnamed_addr constant [53 x i8] c"void NS::Base::functionTemplate1(T) [T = NS::Base *]\00"
// CHECK-DAG: private unnamed_addr constant [46 x i8] c"void NS::Base::functionTemplate1(T) [T = int]\00"

// CHECK-DAG: private unnamed_addr constant [23 x i8] c"anonymousUnionFunction\00"
// CHECK-DAG: private unnamed_addr constant [83 x i8] c"void NS::ContainerForAnonymousRecords::(anonymous union)::anonymousUnionFunction()\00"

// CHECK-DAG: private unnamed_addr constant [24 x i8] c"anonymousStructFunction\00"
// CHECK-DAG: private unnamed_addr constant [85 x i8] c"void NS::ContainerForAnonymousRecords::(anonymous struct)::anonymousStructFunction()\00"

// CHECK-DAG: private unnamed_addr constant [23 x i8] c"anonymousClassFunction\00"
// CHECK-DAG: private unnamed_addr constant [83 x i8] c"void NS::ContainerForAnonymousRecords::(anonymous class)::anonymousClassFunction()\00"

// CHECK-DAG: private unnamed_addr constant [12 x i8] c"~Destructor\00"
// CHECK-DAG: private unnamed_addr constant [30 x i8] c"NS::Destructor::~Destructor()\00"

// CHECK-DAG: private unnamed_addr constant [12 x i8] c"Constructor\00"
// CHECK-DAG: private unnamed_addr constant [41 x i8] c"NS::Constructor::Constructor(NS::Base *)\00"
// CHECK-DAG: private unnamed_addr constant [34 x i8] c"NS::Constructor::Constructor(int)\00"
// CHECK-DAG: private unnamed_addr constant [31 x i8] c"NS::Constructor::Constructor()\00"

// CHECK-DAG: private unnamed_addr constant [16 x i8] c"virtualFunction\00"
// CHECK-DAG: private unnamed_addr constant [44 x i8] c"virtual void NS::Derived::virtualFunction()\00"

// CHECK-DAG: private unnamed_addr constant [21 x i8] c"refQualifiedFunction\00"
// CHECK-DAG: private unnamed_addr constant [41 x i8] c"void NS::Base::refQualifiedFunction() &&\00"
// CHECK-DAG: private unnamed_addr constant [40 x i8] c"void NS::Base::refQualifiedFunction() &\00"

// CHECK-DAG: private unnamed_addr constant [22 x i8] c"constVolatileFunction\00"
// CHECK-DAG: private unnamed_addr constant [54 x i8] c"void NS::Base::constVolatileFunction() const volatile\00"

// CHECK-DAG: private unnamed_addr constant [17 x i8] c"volatileFunction\00"
// CHECK-DAG: private unnamed_addr constant [43 x i8] c"void NS::Base::volatileFunction() volatile\00"

// CHECK-DAG: private unnamed_addr constant [14 x i8] c"constFunction\00"
// CHECK-DAG: private unnamed_addr constant [37 x i8] c"void NS::Base::constFunction() const\00"

// CHECK-DAG: private unnamed_addr constant [26 x i8] c"functionReturingTemplate2\00"
// CHECK-DAG: private unnamed_addr constant [64 x i8] c"ClassTemplate<NS::Base *> NS::Base::functionReturingTemplate2()\00"

// CHECK-DAG: private unnamed_addr constant [26 x i8] c"functionReturingTemplate1\00"
// CHECK-DAG: private unnamed_addr constant [57 x i8] c"ClassTemplate<int> NS::Base::functionReturingTemplate1()\00"

// CHECK-DAG: private unnamed_addr constant [23 x i8] c"withTemplateParameter2\00"
// CHECK-DAG: private unnamed_addr constant [65 x i8] c"void NS::Base::withTemplateParameter2(ClassTemplate<NS::Base *>)\00"

// CHECK-DAG: private unnamed_addr constant [23 x i8] c"withTemplateParameter1\00"
// CHECK-DAG: private unnamed_addr constant [58 x i8] c"void NS::Base::withTemplateParameter1(ClassTemplate<int>)\00"

// CHECK-DAG: private unnamed_addr constant [23 x i8] c"functionReturningClass\00"
// CHECK-DAG: private unnamed_addr constant [45 x i8] c"NS::Base *NS::Base::functionReturningClass()\00"

// CHECK-DAG: private unnamed_addr constant [23 x i8] c"functionWithParameters\00"
// CHECK-DAG: private unnamed_addr constant [64 x i8] c"void NS::Base::functionWithParameters(int, float *, NS::Base *)\00"

// CHECK-DAG: private unnamed_addr constant [17 x i8] c"variadicFunction\00"
// CHECK-DAG: private unnamed_addr constant [42 x i8] c"void NS::Base::variadicFunction(int, ...)\00"

// CHECK-DAG: private unnamed_addr constant [41 x i8] c"virtual void NS::Base::virtualFunction()\00"

// CHECK-DAG: private unnamed_addr constant [15 x i8] c"inlineFunction\00"
// CHECK-DAG: private unnamed_addr constant [32 x i8] c"void NS::Base::inlineFunction()\00"

// CHECK-DAG: private unnamed_addr constant [15 x i8] c"staticFunction\00"
// CHECK-DAG: private unnamed_addr constant [39 x i8] c"static void NS::Base::staticFunction()\00"

// CHECK-DAG: private unnamed_addr constant [26 x i8] c"topLevelNamespaceFunction\00"
// CHECK-DAG: private unnamed_addr constant [59 x i8] c"void ClassInTopLevelNamespace::topLevelNamespaceFunction()\00"

// CHECK-DAG: private unnamed_addr constant [27 x i8] c"anonymousNamespaceFunction\00"
// CHECK-DAG: private unnamed_addr constant [84 x i8] c"void (anonymous namespace)::ClassInAnonymousNamespace::anonymousNamespaceFunction()\00"

// CHECK-DAG: private unnamed_addr constant [19 x i8] c"localClassFunction\00"
// CHECK-DAG: private unnamed_addr constant [59 x i8] c"void NS::localClass(int)::LocalClass::localClassFunction()\00"



int printf(const char * _Format, ...);

class ClassInTopLevelNamespace {
public:
  void topLevelNamespaceFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

namespace {

  class ClassInAnonymousNamespace {
  public:
    void anonymousNamespaceFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  };

} // end anonymous namespace

namespace NS {

template<typename T>
class ClassTemplate {
public:
  void classTemplateFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

class Base {
public:
  static void staticFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  inline void (inlineFunction)() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  virtual void virtualFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void functionWithParameters(int, float*, Base* base) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  Base *functionReturningClass() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return 0;
  }

  void variadicFunction(int, ...) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void withTemplateParameter1(ClassTemplate<int>) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void withTemplateParameter2(ClassTemplate<Base *>) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  ClassTemplate<int> functionReturingTemplate1() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return ClassTemplate<int>();
  }

  ClassTemplate<Base *> functionReturingTemplate2() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    return ClassTemplate<Base *>();
  }

  template<typename T>
  void functionTemplate1(T t) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void constFunction() const {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void volatileFunction() volatile {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
  
  void constVolatileFunction() const volatile {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void refQualifiedFunction() & {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  void refQualifiedFunction() && {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

class Derived : public Base {
public:
  // Virtual function without being explicitly written.
  void virtualFunction() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

class Constructor {
public:
  Constructor() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  Constructor(int) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }

  Constructor(Base *) {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

class Destructor {
public:
  ~Destructor() {
    printf("__func__ %s\n", __func__);
    printf("__FUNCTION__ %s\n", __FUNCTION__);
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

class ContainerForAnonymousRecords {
public:
  class {
  public:
    void anonymousClassFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  } anonymousClass;

  struct {
    void anonymousStructFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  } anonymousStruct;

  union {
    void anonymousUnionFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  } anonymousUnion;
};

void localClass(int) {
  class LocalClass {
  public:
    void localClassFunction() {
      printf("__func__ %s\n", __func__);
      printf("__FUNCTION__ %s\n", __FUNCTION__);
      printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
    }
  };
  LocalClass lc;
  lc.localClassFunction();
}

extern void externFunction() {
  printf("__func__ %s\n", __func__);
  printf("__FUNCTION__ %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

} // end NS namespace

// additional tests for __PRETTY_FUNCTION__
template <typename T, typename U>
void functionTemplateWithTwoParams(T, U)
{
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

template <typename T>
void functionTemplateWithoutParameterList()
{
  T t = T();

  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

template <typename T>
T functionTemplateWithTemplateReturnType()
{
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);

  return T();
}

template <typename T>
T * functionTemplateWithCompoundTypes(T a[])
{
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);

  return 0;
}

template <typename T>
void functionTemplateExplicitSpecialization(T t)
{
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

template <>
void functionTemplateExplicitSpecialization<int>(int i)
{
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

template <typename, typename T>
void functionTemplateWithUnnamedTemplateParameter(T t)
{
  printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
}

template <typename T>
void functionTemplateWithLambda(T t)
{
  []() {
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  } ();
}

template <typename T>
void functionTemplateWithCapturedStmt(T t)
{
  #pragma clang __debug captured
  {
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
}

template <typename T>
class OuterClass
{
public:
  class MiddleClass
  {
  public:
    template <typename U>
    class InnerClass
    {
    public:
      void memberFunction(T x, U y) const
      {
        printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
      }
    };
  };
};

template <typename T, template <typename> class Param = NS::ClassTemplate>
class ClassWithTemplateTemplateParam
{
public:
  static void staticMember()
  {
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

template <int Count>
class NonTypeTemplateParam
{
public:
  void size() const
  {
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

template <typename T>
class SpecializedClassTemplate
{
public:
  template <typename U>
  void memberFunctionTemplate(T t, U u) const
  {
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

template <>
class SpecializedClassTemplate<int>
{
public:
  template <typename U>
  void memberFunctionTemplate(int i, U u) const
  {
    printf("__PRETTY_FUNCTION__ %s\n\n", __PRETTY_FUNCTION__);
  }
};

int main() {
  ClassInAnonymousNamespace anonymousNamespace;
  anonymousNamespace.anonymousNamespaceFunction();

  ClassInTopLevelNamespace topLevelNamespace;
  topLevelNamespace.topLevelNamespaceFunction();

  NS::Base::staticFunction();
  
  NS::Base b;
  b.inlineFunction();
  b.virtualFunction();
  b.variadicFunction(0);
  b.functionWithParameters(0, 0, 0);
  b.functionReturningClass();
  
  b.withTemplateParameter1(NS::ClassTemplate<int>());
  b.withTemplateParameter2(NS::ClassTemplate<NS::Base *>());
  b.functionReturingTemplate1();
  b.functionReturingTemplate2();
  b.functionTemplate1<int>(0);
  b.functionTemplate1<NS::Base *>(0);
  b.constFunction();
  b.volatileFunction();
  b.constVolatileFunction();
  b.refQualifiedFunction();
  NS::Base().refQualifiedFunction();

  NS::Derived d;
  d.virtualFunction();
  
  NS::ClassTemplate<int> t1;
  t1.classTemplateFunction();
  NS::ClassTemplate<NS::Base *> t2;
  t2.classTemplateFunction();
  
  NS::Constructor c1;
  NS::Constructor c2(0);
  NS::Constructor c3((NS::Base *)0);
  
  {
    NS::Destructor destructor;
  }

  NS::ContainerForAnonymousRecords anonymous; 
  anonymous.anonymousClass.anonymousClassFunction();
  anonymous.anonymousStruct.anonymousStructFunction();
  anonymous.anonymousUnion.anonymousUnionFunction();

  NS::localClass(0);

  NS::externFunction();

  // additional tests for __PRETTY_FUNCTION__

  functionTemplateWithTwoParams(0, 0.0f);
  functionTemplateWithoutParameterList<double>();
  functionTemplateWithTemplateReturnType<char>();
  int array[] = { 1, 2, 3 };
  functionTemplateWithCompoundTypes(array);
  functionTemplateExplicitSpecialization(0);
  functionTemplateExplicitSpecialization(0.0);
  functionTemplateWithUnnamedTemplateParameter<int, float>(0.0f);

  functionTemplateWithLambda<int>(0);
  functionTemplateWithCapturedStmt<int>(0);

  OuterClass<int *>::MiddleClass::InnerClass<float> omi;
  omi.memberFunction(0, 0.0f);

  ClassWithTemplateTemplateParam<char>::staticMember();

  NonTypeTemplateParam<42> ntt;
  ntt.size();

  SpecializedClassTemplate<int> sct1;
  sct1.memberFunctionTemplate(0, 0.0f);
  SpecializedClassTemplate<char> sct2;
  sct2.memberFunctionTemplate('0', 0.0);

  return 0;
}

// rdar://19065361
class XXX {
  XXX();
  ~XXX();
};

void XXLog(const char *functionName) { }

typedef void (^notify_handler_t)(int token);

typedef void (^dispatch_block_t)(void);

void notify_register_dispatch(notify_handler_t handler);

void _dispatch_once(dispatch_block_t block);

XXX::XXX()
{
   _dispatch_once(^{ notify_register_dispatch( ^(int token) { XXLog(__FUNCTION__); }); 
   });
}
// CHECK: define internal {{.*}}void @___ZN3XXXC2Ev_block_invoke_

XXX::~XXX()
{
   _dispatch_once(^{ notify_register_dispatch( ^(int token) { XXLog(__FUNCTION__); }); 
   });
}
// CHECK: define internal {{.*}}void @___ZN3XXXD2Ev_block_invoke_

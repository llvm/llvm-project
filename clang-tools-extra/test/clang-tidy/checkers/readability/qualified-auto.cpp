// RUN: %check_clang_tidy %s readability-qualified-auto %t

namespace typedefs {
typedef int *MyPtr;
typedef int &MyRef;
typedef const int *CMyPtr;
typedef const int &CMyRef;

MyPtr getPtr();
MyRef getRef();
CMyPtr getCPtr();
CMyRef getCRef();

void foo() {
  auto TdNakedPtr = getPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto TdNakedPtr' can be declared as 'auto *TdNakedPtr'
  // CHECK-FIXES: {{^}}  auto *TdNakedPtr = getPtr();
  auto &TdNakedRef = getRef();
  auto TdNakedRefDeref = getRef();
  auto TdNakedCPtr = getCPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto TdNakedCPtr' can be declared as 'const auto *TdNakedCPtr'
  // CHECK-FIXES: {{^}}  const auto *TdNakedCPtr = getCPtr();
  auto &TdNakedCRef = getCRef();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto &TdNakedCRef' can be declared as 'const auto &TdNakedCRef'
  // CHECK-FIXES: {{^}}  const auto &TdNakedCRef = getCRef();
  auto TdNakedCRefDeref = getCRef();
}

}; // namespace typedefs

namespace usings {
using MyPtr = int *;
using MyRef = int &;
using CMyPtr = const int *;
using CMyRef = const int &;

MyPtr getPtr();
MyRef getRef();
CMyPtr getCPtr();
CMyRef getCRef();

void foo() {
  auto UNakedPtr = getPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto UNakedPtr' can be declared as 'auto *UNakedPtr'
  // CHECK-FIXES: {{^}}  auto *UNakedPtr = getPtr();
  auto &UNakedRef = getRef();
  auto UNakedRefDeref = getRef();
  auto UNakedCPtr = getCPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto UNakedCPtr' can be declared as 'const auto *UNakedCPtr'
  // CHECK-FIXES: {{^}}  const auto *UNakedCPtr = getCPtr();
  auto &UNakedCRef = getCRef();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto &UNakedCRef' can be declared as 'const auto &UNakedCRef'
  // CHECK-FIXES: {{^}}  const auto &UNakedCRef = getCRef();
  auto UNakedCRefDeref = getCRef();
}

}; // namespace usings

int getInt();
int *getIntPtr();
const int *getCIntPtr();

void foo() {
  // make sure check disregards named types
  int TypedInt = getInt();
  int *TypedPtr = getIntPtr();
  const int *TypedConstPtr = getCIntPtr();
  int &TypedRef = *getIntPtr();
  const int &TypedConstRef = *getCIntPtr();

  // make sure check disregards auto types that aren't pointers or references
  auto AutoInt = getInt();

  auto NakedPtr = getIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto NakedPtr' can be declared as 'auto *NakedPtr'
  // CHECK-FIXES: {{^}}  auto *NakedPtr = getIntPtr();
  auto NakedCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto NakedCPtr' can be declared as 'const auto *NakedCPtr'
  // CHECK-FIXES: {{^}}  const auto *NakedCPtr = getCIntPtr();

  const auto ConstPtr = getIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'const auto ConstPtr' can be declared as 'auto *const ConstPtr'
  // CHECK-FIXES: {{^}}  auto *const ConstPtr = getIntPtr();
  const auto ConstCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'const auto ConstCPtr' can be declared as 'const auto *const ConstCPtr'
  // CHECK-FIXES: {{^}}  const auto *const ConstCPtr = getCIntPtr();

  volatile auto VolatilePtr = getIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'volatile auto VolatilePtr' can be declared as 'auto *volatile VolatilePtr'
  // CHECK-FIXES: {{^}}  auto *volatile VolatilePtr = getIntPtr();
  volatile auto VolatileCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'volatile auto VolatileCPtr' can be declared as 'const auto *volatile VolatileCPtr'
  // CHECK-FIXES: {{^}}  const auto *volatile VolatileCPtr = getCIntPtr();

  auto *QualPtr = getIntPtr();
  auto *QualCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto *QualCPtr' can be declared as 'const auto *QualCPtr'
  // CHECK-FIXES: {{^}}  const auto *QualCPtr = getCIntPtr();
  auto *const ConstantQualCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto *const ConstantQualCPtr' can be declared as 'const auto *const ConstantQualCPtr'
  // CHECK-FIXES: {{^}}  const auto *const ConstantQualCPtr = getCIntPtr();
  auto *volatile VolatileQualCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto *volatile VolatileQualCPtr' can be declared as 'const auto *volatile VolatileQualCPtr'
  // CHECK-FIXES: {{^}}  const auto *volatile VolatileQualCPtr = getCIntPtr();
  const auto *ConstQualCPtr = getCIntPtr();

  auto &Ref = *getIntPtr();
  auto &CRef = *getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto &CRef' can be declared as 'const auto &CRef'
  // CHECK-FIXES: {{^}}  const auto &CRef = *getCIntPtr();
  const auto &ConstCRef = *getCIntPtr();

  if (auto X = getCIntPtr()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES: {{^}}  if (const auto *X = getCIntPtr()) {
  }
}

void macroTest() {
#define _AUTO auto
#define _CONST const
  _AUTO AutoMACROPtr = getIntPtr();
  const _AUTO ConstAutoMacroPtr = getIntPtr();
  _CONST _AUTO ConstMacroAutoMacroPtr = getIntPtr();
  _CONST auto ConstMacroAutoPtr = getIntPtr();
#undef _AUTO
#undef _CONST
}

namespace std {
template <typename T>
class vector { // dummy impl
  T _data[1];

public:
  T *begin() { return _data; }
  const T *begin() const { return _data; }
  T *end() { return &_data[1]; }
  const T *end() const { return &_data[1]; }
};
} // namespace std

void change(int &);
void observe(const int &);

void loopRef(std::vector<int> &Mutate, const std::vector<int> &Constant) {
  for (auto &Data : Mutate) {
    change(Data);
  }
  for (auto &Data : Constant) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto &Data' can be declared as 'const auto &Data'
    // CHECK-FIXES: {{^}}  for (const auto &Data : Constant) {
    observe(Data);
  }
}

void loopPtr(const std::vector<int *> &Mutate, const std::vector<const int *> &Constant) {
  for (auto Data : Mutate) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto Data' can be declared as 'auto *Data'
    // CHECK-FIXES: {{^}}  for (auto *Data : Mutate) {
    change(*Data);
  }
  for (auto Data : Constant) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto Data' can be declared as 'const auto *Data'
    // CHECK-FIXES: {{^}}  for (const auto *Data : Constant) {
    observe(*Data);
  }
}

template <typename T>
void tempLoopPtr(std::vector<T *> &MutateTemplate, std::vector<const T *> &ConstantTemplate) {
  for (auto Data : MutateTemplate) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto Data' can be declared as 'auto *Data'
    // CHECK-FIXES: {{^}}  for (auto *Data : MutateTemplate) {
    change(*Data);
  }
  //FixMe
  for (auto Data : ConstantTemplate) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto Data' can be declared as 'const auto *Data'
    // CHECK-FIXES: {{^}}  for (const auto *Data : ConstantTemplate) {
    observe(*Data);
  }
}

template <typename T>
class TemplateLoopPtr {
public:
  void operator()(const std::vector<T *> &MClassTemplate, const std::vector<const T *> &CClassTemplate) {
    for (auto Data : MClassTemplate) {
      // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'auto Data' can be declared as 'auto *Data'
      // CHECK-FIXES: {{^}}    for (auto *Data : MClassTemplate) {
      change(*Data);
    }
    //FixMe
    for (auto Data : CClassTemplate) {
      // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'auto Data' can be declared as 'const auto *Data'
      // CHECK-FIXES: {{^}}    for (const auto *Data : CClassTemplate) {
      observe(*Data);
    }
  }
};

void bar() {
  std::vector<int> Vec;
  std::vector<int *> PtrVec;
  std::vector<const int *> CPtrVec;
  loopRef(Vec, Vec);
  loopPtr(PtrVec, CPtrVec);
  tempLoopPtr(PtrVec, CPtrVec);
  TemplateLoopPtr<int>()(PtrVec, CPtrVec);
}

typedef int *(*functionRetPtr)();
typedef int (*functionRetVal)();

functionRetPtr getPtrFunction();
functionRetVal getValFunction();

void baz() {
  auto MyFunctionPtr = getPtrFunction();
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto MyFunctionPtr' can be declared as 'auto *MyFunctionPtr'
  // CHECK-FIXES-NOT: {{^}}  auto *MyFunctionPtr = getPtrFunction();
  auto MyFunctionVal = getValFunction();
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto MyFunctionVal' can be declared as 'auto *MyFunctionVal'
  // CHECK-FIXES-NOT: {{^}}  auto *MyFunctionVal = getValFunction();

  auto LambdaTest = [] { return 0; };
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto LambdaTest' can be declared as 'auto *LambdaTest'
  // CHECK-FIXES-NOT: {{^}}  auto *LambdaTest = [] { return 0; };

  auto LambdaTest2 = +[] { return 0; };
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto LambdaTest2' can be declared as 'auto *LambdaTest2'
  // CHECK-FIXES-NOT: {{^}}  auto *LambdaTest2 = +[] { return 0; };

  auto MyFunctionRef = *getPtrFunction();
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto MyFunctionRef' can be declared as 'auto *MyFunctionRef'
  // CHECK-FIXES-NOT: {{^}}  auto *MyFunctionRef = *getPtrFunction();

  auto &MyFunctionRef2 = *getPtrFunction();
}

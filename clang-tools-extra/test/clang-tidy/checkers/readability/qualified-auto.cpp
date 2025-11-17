// RUN: %check_clang_tidy %s readability-qualified-auto %t \
// RUN: -config='{CheckOptions: { \
// RUN:   readability-qualified-auto.AllowedTypes: "[iI]terator$;my::ns::Ignored1;std::array<.*>::Ignored2;MyIgnoredPtr" \
// RUN: }}'
// RUN: %check_clang_tidy %s readability-qualified-auto %t \
// RUN: -config='{CheckOptions: { \
// RUN:   readability-qualified-auto.AllowedTypes: "[iI]terator$;my::ns::Ignored1;std::array<.*>::Ignored2;MyIgnoredPtr", \
// RUN:   readability-qualified-auto.IgnoreAliasing: false \
// RUN: }}' -check-suffix=ALIAS -- 

namespace typedefs {
typedef int *MyPtr;
typedef int &MyRef;
typedef const int *CMyPtr;
typedef const int &CMyRef;

MyPtr getPtr();
MyPtr* getPtrPtr();
MyRef getRef();
CMyPtr getCPtr();
CMyPtr* getCPtrPtr();
CMyRef getCRef();

void foo() {
  auto TdNakedPtr = getPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto TdNakedPtr' can be declared as 'auto *TdNakedPtr'
  // CHECK-FIXES: auto *TdNakedPtr = getPtr();
  auto TdNakedPtrPtr = getPtrPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto TdNakedPtrPtr' can be declared as 'auto *TdNakedPtrPtr'
  // CHECK-FIXES: auto *TdNakedPtrPtr = getPtrPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto TdNakedPtrPtr' can be declared as 'auto *TdNakedPtrPtr'
  // CHECK-FIXES-ALIAS: auto *TdNakedPtrPtr = getPtrPtr();
  auto &TdNakedRef = getRef();
  auto TdNakedRefDeref = getRef();
  auto TdNakedCPtr = getCPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto TdNakedCPtr' can be declared as 'const auto *TdNakedCPtr'
  // CHECK-FIXES: const auto *TdNakedCPtr = getCPtr();
  auto TdNakedCPtrPtr = getCPtrPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto TdNakedCPtrPtr' can be declared as 'auto *TdNakedCPtrPtr'
  // CHECK-FIXES: auto *TdNakedCPtrPtr = getCPtrPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto TdNakedCPtrPtr' can be declared as 'auto *TdNakedCPtrPtr'
  // CHECK-FIXES-ALIAS: auto *TdNakedCPtrPtr = getCPtrPtr();
  auto &TdNakedCRef = getCRef();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto &TdNakedCRef' can be declared as 'const auto &TdNakedCRef'
  // CHECK-FIXES: const auto &TdNakedCRef = getCRef();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto &TdNakedCRef' can be declared as 'const auto &TdNakedCRef'
  // CHECK-FIXES-ALIAS: const auto &TdNakedCRef = getCRef();
  auto TdNakedCRefDeref = getCRef();
}

}; // namespace typedefs

namespace usings {
using MyPtr = int *;
using MyRef = int &;
using CMyPtr = const int *;
using CMyRef = const int &;

MyPtr getPtr();
MyPtr* getPtrPtr();
MyRef getRef();
CMyPtr getCPtr();
CMyRef getCRef();

void foo() {
  auto UNakedPtr = getPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto UNakedPtr' can be declared as 'auto *UNakedPtr'
  // CHECK-FIXES: auto *UNakedPtr = getPtr();
  auto UNakedPtrPtr = getPtrPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto UNakedPtrPtr' can be declared as 'auto *UNakedPtrPtr'
  // CHECK-FIXES: auto *UNakedPtrPtr = getPtrPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto UNakedPtrPtr' can be declared as 'auto *UNakedPtrPtr'
  // CHECK-FIXES-ALIAS: auto *UNakedPtrPtr = getPtrPtr();
  auto &UNakedRef = getRef();
  auto UNakedRefDeref = getRef();
  auto UNakedCPtr = getCPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto UNakedCPtr' can be declared as 'const auto *UNakedCPtr'
  // CHECK-FIXES: const auto *UNakedCPtr = getCPtr();
  auto &UNakedCRef = getCRef();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto &UNakedCRef' can be declared as 'const auto &UNakedCRef'
  // CHECK-FIXES: const auto &UNakedCRef = getCRef();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto &UNakedCRef' can be declared as 'const auto &UNakedCRef'
  // CHECK-FIXES-ALIAS: const auto &UNakedCRef = getCRef();
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
  // CHECK-FIXES: auto *NakedPtr = getIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto NakedPtr' can be declared as 'auto *NakedPtr'
  // CHECK-FIXES-ALIAS: auto *NakedPtr = getIntPtr();
  auto NakedCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto NakedCPtr' can be declared as 'const auto *NakedCPtr'
  // CHECK-FIXES: const auto *NakedCPtr = getCIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto NakedCPtr' can be declared as 'const auto *NakedCPtr'
  // CHECK-FIXES-ALIAS: const auto *NakedCPtr = getCIntPtr();

  const auto ConstPtr = getIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'const auto ConstPtr' can be declared as 'auto *const ConstPtr'
  // CHECK-FIXES: auto *const ConstPtr = getIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'const auto ConstPtr' can be declared as 'auto *const ConstPtr'
  // CHECK-FIXES-ALIAS: auto *const ConstPtr = getIntPtr();
  const auto ConstCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'const auto ConstCPtr' can be declared as 'const auto *const ConstCPtr'
  // CHECK-FIXES: const auto *const ConstCPtr = getCIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'const auto ConstCPtr' can be declared as 'const auto *const ConstCPtr'
  // CHECK-FIXES-ALIAS: const auto *const ConstCPtr = getCIntPtr();

  volatile auto VolatilePtr = getIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'volatile auto VolatilePtr' can be declared as 'auto *volatile VolatilePtr'
  // CHECK-FIXES: auto *volatile VolatilePtr = getIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'volatile auto VolatilePtr' can be declared as 'auto *volatile VolatilePtr'
  // CHECK-FIXES-ALIAS: auto *volatile VolatilePtr = getIntPtr();
  volatile auto VolatileCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'volatile auto VolatileCPtr' can be declared as 'const auto *volatile VolatileCPtr'
  // CHECK-FIXES: const auto *volatile VolatileCPtr = getCIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'volatile auto VolatileCPtr' can be declared as 'const auto *volatile VolatileCPtr'
  // CHECK-FIXES-ALIAS: const auto *volatile VolatileCPtr = getCIntPtr();

  auto *QualPtr = getIntPtr();
  auto *QualCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto *QualCPtr' can be declared as 'const auto *QualCPtr'
  // CHECK-FIXES: const auto *QualCPtr = getCIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto *QualCPtr' can be declared as 'const auto *QualCPtr'
  // CHECK-FIXES-ALIAS: const auto *QualCPtr = getCIntPtr();
  auto *const ConstantQualCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto *const ConstantQualCPtr' can be declared as 'const auto *const ConstantQualCPtr'
  // CHECK-FIXES: const auto *const ConstantQualCPtr = getCIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto *const ConstantQualCPtr' can be declared as 'const auto *const ConstantQualCPtr'
  // CHECK-FIXES-ALIAS: const auto *const ConstantQualCPtr = getCIntPtr();
  auto *volatile VolatileQualCPtr = getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto *volatile VolatileQualCPtr' can be declared as 'const auto *volatile VolatileQualCPtr'
  // CHECK-FIXES: const auto *volatile VolatileQualCPtr = getCIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto *volatile VolatileQualCPtr' can be declared as 'const auto *volatile VolatileQualCPtr'
  // CHECK-FIXES-ALIAS: const auto *volatile VolatileQualCPtr = getCIntPtr();
  const auto *ConstQualCPtr = getCIntPtr();

  auto &Ref = *getIntPtr();
  auto &CRef = *getCIntPtr();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto &CRef' can be declared as 'const auto &CRef'
  // CHECK-FIXES: const auto &CRef = *getCIntPtr();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto &CRef' can be declared as 'const auto &CRef'
  // CHECK-FIXES-ALIAS: const auto &CRef = *getCIntPtr();
  const auto &ConstCRef = *getCIntPtr();

  if (auto X = getCIntPtr()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES: if (const auto *X = getCIntPtr()) {
    // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:7: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES-ALIAS: if (const auto *X = getCIntPtr()) {
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
    // CHECK-FIXES: for (const auto &Data : Constant) {
    // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:8: warning: 'auto &Data' can be declared as 'const auto &Data'
    // CHECK-FIXES-ALIAS: for (const auto &Data : Constant) {
    observe(Data);
  }
}

void loopPtr(const std::vector<int *> &Mutate, const std::vector<const int *> &Constant) {
  for (auto Data : Mutate) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto Data' can be declared as 'auto *Data'
    // CHECK-FIXES: for (auto *Data : Mutate) {
    // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:8: warning: 'auto Data' can be declared as 'auto *Data'
    // CHECK-FIXES-ALIAS: for (auto *Data : Mutate) {
    change(*Data);
  }
  for (auto Data : Constant) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto Data' can be declared as 'const auto *Data'
    // CHECK-FIXES: for (const auto *Data : Constant) {
    // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:8: warning: 'auto Data' can be declared as 'const auto *Data'
    // CHECK-FIXES-ALIAS: for (const auto *Data : Constant) {
    observe(*Data);
  }
}

template <typename T>
void tempLoopPtr(std::vector<T *> &MutateTemplate, std::vector<const T *> &ConstantTemplate) {
  for (auto Data : MutateTemplate) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto Data' can be declared as 'auto *Data'
    // CHECK-FIXES: for (auto *Data : MutateTemplate) {
    // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:8: warning: 'auto Data' can be declared as 'auto *Data'
    // CHECK-FIXES-ALIAS: for (auto *Data : MutateTemplate) {
    change(*Data);
  }
  //FixMe
  for (auto Data : ConstantTemplate) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto Data' can be declared as 'const auto *Data'
    // CHECK-FIXES: for (const auto *Data : ConstantTemplate) {
    // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:8: warning: 'auto Data' can be declared as 'const auto *Data'
    // CHECK-FIXES-ALIAS: for (const auto *Data : ConstantTemplate) {
    observe(*Data);
  }
}

template <typename T>
class TemplateLoopPtr {
public:
  void operator()(const std::vector<T *> &MClassTemplate, const std::vector<const T *> &CClassTemplate) {
    for (auto Data : MClassTemplate) {
      // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'auto Data' can be declared as 'auto *Data'
      // CHECK-FIXES: for (auto *Data : MClassTemplate) {
      // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:10: warning: 'auto Data' can be declared as 'auto *Data'
      // CHECK-FIXES-ALIAS: for (auto *Data : MClassTemplate) {
      change(*Data);
    }
    //FixMe
    for (auto Data : CClassTemplate) {
      // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'auto Data' can be declared as 'const auto *Data'
      // CHECK-FIXES: for (const auto *Data : CClassTemplate) {
      // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:10: warning: 'auto Data' can be declared as 'const auto *Data'
      // CHECK-FIXES-ALIAS: for (const auto *Data : CClassTemplate) {
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
  // CHECK-FIXES-NOT: auto *MyFunctionPtr = getPtrFunction();
  // CHECK-MESSAGES-NOT-ALIAS: :[[@LINE-1]]:3: warning: 'auto MyFunctionPtr' can be declared as 'auto *MyFunctionPtr'
  // CHECK-FIXES-NOT-ALIAS: auto *MyFunctionPtr = getPtrFunction();
  auto MyFunctionVal = getValFunction();
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto MyFunctionVal' can be declared as 'auto *MyFunctionVal'
  // CHECK-FIXES-NOT: auto *MyFunctionVal = getValFunction();
  // CHECK-MESSAGES-NOT-ALIAS: :[[@LINE-3]]:3: warning: 'auto MyFunctionVal' can be declared as 'auto *MyFunctionVal'
  // CHECK-FIXES-NOT-ALIAS: auto *MyFunctionVal = getValFunction();

  auto LambdaTest = [] { return 0; };
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto LambdaTest' can be declared as 'auto *LambdaTest'
  // CHECK-FIXES-NOT: auto *LambdaTest = [] { return 0; };
  // CHECK-MESSAGES-NOT-ALIAS: :[[@LINE-3]]:3: warning: 'auto LambdaTest' can be declared as 'auto *LambdaTest'
  // CHECK-FIXES-NOT-ALIAS: auto *LambdaTest = [] { return 0; };

  auto LambdaTest2 = +[] { return 0; };
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto LambdaTest2' can be declared as 'auto *LambdaTest2'
  // CHECK-FIXES-NOT: auto *LambdaTest2 = +[] { return 0; };
  // CHECK-MESSAGES-NOT-ALIAS: :[[@LINE-3]]:3: warning: 'auto LambdaTest2' can be declared as 'auto *LambdaTest2'
  // CHECK-FIXES-NOT-ALIAS: auto *LambdaTest2 = +[] { return 0; };

  auto MyFunctionRef = *getPtrFunction();
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: 'auto MyFunctionRef' can be declared as 'auto *MyFunctionRef'
  // CHECK-FIXES-NOT: auto *MyFunctionRef = *getPtrFunction();
  // CHECK-MESSAGES-NOT-ALIAS: :[[@LINE-3]]:3: warning: 'auto MyFunctionRef' can be declared as 'auto *MyFunctionRef'
  // CHECK-FIXES-NOT-ALIAS: auto *MyFunctionRef = *getPtrFunction();

  auto &MyFunctionRef2 = *getPtrFunction();
}

namespace std {

template<typename T, int N>
struct array {
  typedef T value_type;

  typedef value_type* iterator;
  typedef value_type* Iterator;
  using using_iterator = T*;
  typedef const value_type* const_iterator;
  typedef const value_type* constIterator;

  struct Ignored2 {};
  using NotIgnored2 = Ignored2;

  iterator begin() { return nullptr; }
  const_iterator begin() const { return nullptr; }
  iterator end() { return nullptr; }
  const_iterator end() const { return nullptr; }
};

struct Iterator {};

struct Ignored2 {}; // should not be ignored

} // namespace std

typedef std::Iterator iterator;

namespace my {
namespace ns {

struct Ignored1 {};

using NotIgnored1 = Ignored1;
typedef Ignored1 NotIgnored2;

} // namespace ns

struct Ignored1 {}; // should not be ignored

} // namespace my

typedef int *MyIgnoredPtr;
MyIgnoredPtr getIgnoredPtr();

void ignored_types() {
  auto ignored_ptr = getIgnoredPtr();
  // CHECK-MESSAGES-NOT: warning: 'auto ignored_ptr' can be declared as 'auto *ignored_ptr'
  // CHECK-FIXES-NOT: auto *ignored_ptr = getIgnoredPtr();

  std::array<int, 4> arr;
  std::array<int, 4> carr;

  auto it1 = arr.begin();
  // CHECK-MESSAGES-NOT: warning: 'auto it' can be declared as 'auto *it'
  // CHECK-FIXES-NOT: auto *it = vec.it_begin();
  
  auto it2 = carr.begin();
  // CHECK-MESSAGES-NOT: warning: 'auto it2' can be declared as 'auto *it2'
  // CHECK-FIXES-NOT: auto *it2 = carr.begin();

  auto it3 = std::array<int, 4>::iterator{};
  // CHECK-MESSAGES-NOT: warning: 'auto it3' can be declared as 'auto *it3'
  // CHECK-FIXES-NOT: auto *it3 = std::array<int, 4>::iterator{};

  auto it4 = std::array<int, 4>::Iterator{};
  // CHECK-MESSAGES-NOT: warning: 'auto it4' can be declared as 'auto *it4'
  // CHECK-FIXES-NOT: auto *it4 = std::array<int, 4>::Iterator{};

  auto it5 = std::array<int, 4>::using_iterator{};
  // CHECK-MESSAGES-NOT: warning: 'auto it5' can be declared as 'auto *it5'
  // CHECK-FIXES-NOT: auto *it5 = std::array<int, 4>::using_iterator{};

  auto it6 = std::array<int, 4>::const_iterator{};
  // CHECK-MESSAGES-NOT: warning: 'auto it6' can be declared as 'auto *it6'
  // CHECK-FIXES-NOT: auto *it6 = std::array<int, 4>::const_iterator{};

  auto it7 = std::array<int, 4>::constIterator{};
  // CHECK-MESSAGES-NOT: warning: 'auto it7' can be declared as 'auto *it7'
  // CHECK-FIXES-NOT: auto *it7 = std::array<int, 4>::constIterator{};

  auto it8 = new std::Iterator();
  // CHECK-MESSAGES-NOT: warning: 'auto it8' can be declared as 'auto *it8'
  // CHECK-FIXES-NOT: auto *it8 = new std::Iterator();

  auto it9 = new iterator();
  // CHECK-MESSAGES-NOT: warning: 'auto it9' can be declared as 'auto *it9'
  // CHECK-FIXES-NOT: auto *it9 = new iterator();

  auto arr_ignored2 = new std::array<int, 4>::Ignored2();
  // CHECK-MESSAGES-NOT: warning: 'auto arr_ignored2' can be declared as 'auto *arr_ignored2'
  // CHECK-FIXES-NOT: auto *arr_ignored2 = new std::array<int, 4>::Ignored2();

  auto arr_not_ignored2 = new std::array<int, 4>::NotIgnored2();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto arr_not_ignored2' can be declared as 'auto *arr_not_ignored2'
  // CHECK-FIXES: auto *arr_not_ignored2 = new std::array<int, 4>::NotIgnored2();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto arr_not_ignored2' can be declared as 'auto *arr_not_ignored2'
  // CHECK-FIXES-ALIAS: auto *arr_not_ignored2 = new std::array<int, 4>::NotIgnored2();

  auto not_ignored2 = new std::Ignored2();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto not_ignored2' can be declared as 'auto *not_ignored2'
  // CHECK-FIXES: auto *not_ignored2 = new std::Ignored2();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto not_ignored2' can be declared as 'auto *not_ignored2'
  // CHECK-FIXES-ALIAS: auto *not_ignored2 = new std::Ignored2();

  auto ignored1 = new my::ns::Ignored1();
  // CHECK-MESSAGES-NOT: warning: 'auto ignored1' can be declared as 'auto *ignored1'
  // CHECK-FIXES-NOT: auto *ignored1 = new my::ns::Ignored1();

  auto not_ignored1 = new my::ns::NotIgnored1();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto not_ignored1' can be declared as 'auto *not_ignored1'
  // CHECK-FIXES: auto *not_ignored1 = new my::ns::NotIgnored1();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto not_ignored1' can be declared as 'auto *not_ignored1'
  // CHECK-FIXES-ALIAS: auto *not_ignored1 = new my::ns::NotIgnored1();

  auto not2_ignored1 = new my::ns::NotIgnored2();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto not2_ignored1' can be declared as 'auto *not2_ignored1'
  // CHECK-FIXES: auto *not2_ignored1 = new my::ns::NotIgnored2();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto not2_ignored1' can be declared as 'auto *not2_ignored1'
  // CHECK-FIXES-ALIAS: auto *not2_ignored1 = new my::ns::NotIgnored2();

  auto not3_ignored1 = new my::Ignored1();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'auto not3_ignored1' can be declared as 'auto *not3_ignored1'
  // CHECK-FIXES: auto *not3_ignored1 = new my::Ignored1();
  // CHECK-MESSAGES-ALIAS: :[[@LINE-3]]:3: warning: 'auto not3_ignored1' can be declared as 'auto *not3_ignored1'
  // CHECK-FIXES-ALIAS: auto *not3_ignored1 = new my::Ignored1();
}

template <typename T>
void ignored_types_template(std::array<T, 4> arr, const std::array<T, 4>& carr) {
  auto it1 = arr.begin();
  // CHECK-MESSAGES-NOT: warning: 'auto it' can be declared as 'auto *it'
  // CHECK-FIXES-NOT: auto *it = arr.it_begin();
  
  auto it2 = carr.begin();
  // CHECK-MESSAGES-NOT: warning: 'auto it2' can be declared as 'auto *it2'
  // CHECK-FIXES-NOT: auto *it2 = carr.begin();

  for (auto Data : arr) {
    // CHECK-MESSAGES-NOT: warning: 'auto Data' can be declared as 'auto *Data'
    // CHECK-FIXES-NOT: for (auto *Data : MClassTemplate) {
    change(*Data);
  }

  for (auto Data : carr) {
    // CHECK-MESSAGES-NOT: warning: 'auto Data' can be declared as 'const auto *Data'
    // CHECK-FIXES-NOT: for (const auto *Data : MClassTemplate) {
    change(*Data);
  }
}

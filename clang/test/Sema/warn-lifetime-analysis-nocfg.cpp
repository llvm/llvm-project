// RUN: %clang_cc1 -fsyntax-only -Wdangling -Wdangling-field -Wreturn-stack-address -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-lifetime-safety -Wexperimental-lifetime-safety -Wno-dangling -verify=cfg %s

#include "Inputs/lifetime-analysis.h"

struct [[gsl::Owner(int)]] MyIntOwner {
  MyIntOwner();
  // TODO: Do this behind a macro and run tests without this dtor to verify trivial dtor cases.
  ~MyIntOwner();
  int &operator*();
};

struct [[gsl::Pointer(int)]] MyIntPointer {
  MyIntPointer(int *p = nullptr);
  // Conversion operator and constructor conversion will result in two
  // different ASTs. The former is tested with another owner and
  // pointer type.
  MyIntPointer(const MyIntOwner &);
  int &operator*();
  MyIntOwner toOwner();
};

struct MySpecialIntPointer : MyIntPointer {
};

// We did see examples in the wild when a derived class changes
// the ownership model. So we have a test for it.
struct [[gsl::Owner(int)]] MyOwnerIntPointer : MyIntPointer {
};

struct [[gsl::Pointer(long)]] MyLongPointerFromConversion {
  MyLongPointerFromConversion(long *p = nullptr);
  long &operator*();
};

struct [[gsl::Owner(long)]] MyLongOwnerWithConversion {
  MyLongOwnerWithConversion();
  operator MyLongPointerFromConversion();
  long &operator*();
  MyIntPointer releaseAsMyPointer();
  long *releaseAsRawPointer();
};

template<class... T> void use(T... arg);

void danglingHeapObject() {
  new MyLongPointerFromConversion(MyLongOwnerWithConversion{}); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  new MyIntPointer(MyIntOwner{}); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
}

void intentionalFalseNegative() {
  int i;
  MyIntPointer p{&i};
  // In this case we do not have enough information in a statement local
  // analysis to detect the problem.
  new MyIntPointer(p);
  new MyIntPointer(MyIntPointer{p});
}

MyIntPointer ownershipTransferToMyPointer() {
  MyLongOwnerWithConversion t;
  return t.releaseAsMyPointer(); // ok
}

long *ownershipTransferToRawPointer() {
  MyLongOwnerWithConversion t;
  return t.releaseAsRawPointer(); // ok
}

struct Y {
  int a[4];
};

void dangligGslPtrFromTemporary() {
  MyIntPointer p = Y{}.a; // TODO
  (void)p;
}

struct DanglingGslPtrField {
  MyIntPointer p; // expected-note {{pointer member declared here}}
  MyLongPointerFromConversion p2; // expected-note {{pointer member declared here}}
  DanglingGslPtrField(int i) : p(&i) {} // TODO
  DanglingGslPtrField() : p2(MyLongOwnerWithConversion{}) {} // expected-warning {{initializing pointer member 'p2' to point to a temporary object whose lifetime is shorter than the lifetime of the constructed object}}
  DanglingGslPtrField(double) : p(MyIntOwner{}) {} // expected-warning {{initializing pointer member 'p' to point to a temporary object whose lifetime is shorter than the lifetime of the constructed object}}
};

MyIntPointer danglingGslPtrFromLocal() {
  int j;
  // Detected only by CFG analysis.
  return &j; // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

MyIntPointer returningLocalPointer() {
  MyIntPointer localPointer;
  return localPointer; // ok
}

MyIntPointer daglingGslPtrFromLocalOwner() {
  MyIntOwner localOwner;
  return localOwner; // expected-warning {{address of stack memory associated with local variable 'localOwner' returned}} \
                     // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

MyLongPointerFromConversion daglingGslPtrFromLocalOwnerConv() {
  MyLongOwnerWithConversion localOwner;
  return localOwner; // expected-warning {{address of stack memory associated with local variable 'localOwner' returned}} \
                     // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

MyIntPointer danglingGslPtrFromTemporary() {
  return MyIntOwner{}; // expected-warning {{returning address of local temporary object}} \
                       // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

MyIntOwner makeTempOwner();

MyIntPointer danglingGslPtrFromTemporary2() {
  return makeTempOwner(); // expected-warning {{returning address of local temporary object}} \
                          // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

MyLongPointerFromConversion danglingGslPtrFromTemporaryConv() {
  return MyLongOwnerWithConversion{}; // expected-warning {{returning address of local temporary object}}
}

int *noFalsePositive(MyIntOwner &o) {
  MyIntPointer p = o;
  return &*p; // ok
}

MyIntPointer global;
MyLongPointerFromConversion global2;

void initLocalGslPtrWithTempOwner() {
  MyIntPointer p = MyIntOwner{}; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                 // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(p);                        // cfg-note {{later used here}}

  MyIntPointer pp = p = MyIntOwner{}; // expected-warning {{object backing the pointer 'p' will be}} \
                                      // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(p, pp);                         // cfg-note {{later used here}}

  p = MyIntOwner{}; // expected-warning {{object backing the pointer 'p' }} \
                    // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(p);           // cfg-note {{later used here}}

  pp = p; // no warning
  use(p, pp);

  global = MyIntOwner{}; // expected-warning {{object backing the pointer 'global' }} \
                         // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(global);           // cfg-note {{later used here}}

  MyLongPointerFromConversion p2 = MyLongOwnerWithConversion{}; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  use(p2);

  p2 = MyLongOwnerWithConversion{}; // expected-warning {{object backing the pointer 'p2' }}
  global2 = MyLongOwnerWithConversion{}; // expected-warning {{object backing the pointer 'global2' }}
  use(global2, p2);
}


struct Unannotated {
  typedef std::vector<int>::iterator iterator;
  iterator begin();
  operator iterator() const;
};

void modelIterators() {
  std::vector<int>::iterator it = std::vector<int>().begin(); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                                              // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  (void)it; // cfg-note {{later used here}}
}

std::vector<int>::iterator modelIteratorReturn() {
  return std::vector<int>().begin(); // expected-warning {{returning address of local temporary object}} \
                                     // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

const int *modelFreeFunctions() {
  return std::data(std::vector<int>()); // expected-warning {{returning address of local temporary object}} \
                                        // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

int &modelAnyCast() {
  return std::any_cast<int&>(std::any{}); // expected-warning {{returning reference to local temporary object}} \
                                          // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

int modelAnyCast2() {
  return std::any_cast<int>(std::any{}); // ok
}

int modelAnyCast3() {
  return std::any_cast<int&>(std::any{}); // ok
}

const char *danglingRawPtrFromLocal() {
  std::basic_string<char> s;
  return s.c_str(); // expected-warning {{address of stack memory associated with local variable 's' returned}} \
                    // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

int &danglingRawPtrFromLocal2() {
  std::optional<int> o;
  return o.value(); // expected-warning {{reference to stack memory associated with local variable 'o' returned}} \
                    // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

int &danglingRawPtrFromLocal3() {
  std::optional<int> o;
  return *o; // expected-warning {{reference to stack memory associated with local variable 'o' returned}} \
             // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

// GH100384
std::string_view containerWithAnnotatedElements() {
  std::string_view c1 = std::vector<std::string>().at(0); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                                          // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(c1);                                                // cfg-note {{later used here}}

  c1 = std::vector<std::string>().at(0); // expected-warning {{object backing the pointer}} \
                                         // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(c1);                               // cfg-note {{later used here}}

  // no warning on constructing from gsl-pointer
  std::string_view c2 = std::vector<std::string_view>().at(0);
  use(c2);

  std::vector<std::string> local;
  return local.at(0); // expected-warning {{address of stack memory associated with local variable}} \
                      // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

std::string_view localUniquePtr(int i) {
  std::unique_ptr<std::string> c1;
  if (i)
    return *c1; // expected-warning {{address of stack memory associated with local variable}} \
                // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
  std::unique_ptr<std::string_view> c2;
  return *c2; // expect no-warning.
}

std::string_view localOptional(int i) {
  std::optional<std::string> o;
  if (i)
    return o.value(); // expected-warning {{address of stack memory associated with local variable}} \
                      // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
  std::optional<std::string_view> abc;
  return abc.value(); // expect no warning
}

const char *danglingRawPtrFromTemp() {
  return std::basic_string<char>().c_str(); // expected-warning {{returning address of local temporary object}} \
                                            // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

std::unique_ptr<int> getUniquePtr();

int *danglingUniquePtrFromTemp() {
  return getUniquePtr().get(); // expected-warning {{returning address of local temporary object}} \
                               // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

int *danglingUniquePtrFromTemp2() {
  return std::unique_ptr<int>().get(); // expected-warning {{returning address of local temporary object}} \
                                       // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

const int& danglingRefToOptionalFromTemp3() {
  return std::optional<int>().value(); // expected-warning {{returning reference to local temporary object}} \
                                       // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

std::optional<std::string> getTempOptStr();

std::string_view danglingRefToOptionalFromTemp4() {
  return getTempOptStr().value(); // expected-warning {{returning address of local temporary object}} \
                                  // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

void danglingReferenceFromTempOwner() {
  int &&r = *std::optional<int>();          // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                            // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  // FIXME: Detect this using the CFG-based lifetime analysis.
  //        https://github.com/llvm/llvm-project/issues/175893
  int &&r2 = *std::optional<int>(5);        // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}

  // FIXME: Detect this using the CFG-based lifetime analysis.
  //        https://github.com/llvm/llvm-project/issues/175893
  int &&r3 = std::optional<int>(5).value(); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}

  const int &r4 = std::vector<int>().at(3); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                            // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  int &&r5 = std::vector<int>().at(3);      // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                            // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(r, r2, r3, r4, r5);                   // cfg-note 3 {{later used here}}

  std::string_view sv = *getTempOptStr();  // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                           // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(sv);                                 // cfg-note {{later used here}}
}

std::vector<int> getTempVec();
std::optional<std::vector<int>> getTempOptVec();

void testLoops() {
  for (auto i : getTempVec()) // ok
    ;
  for (auto i : *getTempOptVec()) // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                  // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}} cfg-note {{later used here}}
    ;
}

int &usedToBeFalsePositive(std::vector<int> &v) {
  std::vector<int>::iterator it = v.begin();
  int& value = *it;
  return value; // ok
}

int &doNotFollowReferencesForLocalOwner() {
// Warning caught by CFG analysis.
  std::unique_ptr<int> localOwner;
  int &p = *localOwner // cfg-warning {{address of stack memory is returned later}}
            .get();
  return p; // cfg-note {{returned here}}
}

const char *trackThroughMultiplePointer() {
  return std::basic_string_view<char>(std::basic_string<char>()).begin(); // expected-warning {{returning address of local temporary object}} \
         // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

struct X {
  X(std::unique_ptr<int> up) :
    pointee(*up), pointee2(up.get()), pointer(std::move(up)) {}
  int &pointee;
  int *pointee2;
  std::unique_ptr<int> pointer;
};

struct [[gsl::Owner]] XOwner {
  int* get() const [[clang::lifetimebound]];
};
struct X2 {
  // A common usage that moves the passing owner to the class.
  // verify no warning on this case.
  X2(XOwner owner) :
    pointee(owner.get()),
    owner(std::move(owner)) {}
  int* pointee;
  XOwner owner;
};

std::vector<int>::iterator getIt();
std::vector<int> getVec();

const int &handleGslPtrInitsThroughReference() {
  const auto &it = getIt(); // Ok, it is lifetime extended.
  return *it;
}

void handleGslPtrInitsThroughReference2() {
  const std::vector<int> &v = getVec();
  const int *val = v.data(); // Ok, it is lifetime extended.
}

void handleTernaryOperator(bool cond) {
    std::basic_string<char> def;
    // FIXME: Detect this using the CFG-based lifetime analysis.
    std::basic_string_view<char> v = cond ? def : ""; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
    use(v);
}

std::string operator+(std::string_view s1, std::string_view s2);
void danglingStringviewAssignment(std::string_view a1, std::string_view a2) {
  a1 = std::string(); // expected-warning {{object backing}} \
                      // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(a1);            // cfg-note {{later used here}}

  a2 = a1 + a1; // expected-warning {{object backing}} \
                // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(a2);      // cfg-note {{later used here}}
}

std::reference_wrapper<int> danglingPtrFromNonOwnerLocal() {
  int i = 5;
  return i; // TODO
}

std::reference_wrapper<int> danglingPtrFromNonOwnerLocal2() {
  int i = 5;
  return std::ref(i); // TODO
}

std::reference_wrapper<int> danglingPtrFromNonOwnerLocal3() {
  int i = 5;
  return std::reference_wrapper<int>(i); // TODO
}

std::reference_wrapper<Unannotated> danglingPtrFromNonOwnerLocal4() {
  Unannotated i;
  return std::reference_wrapper<Unannotated>(i); // TODO
}

std::reference_wrapper<Unannotated> danglingPtrFromNonOwnerLocal5() {
  Unannotated i;
  return std::ref(i); // TODO
}

int *returnPtrToLocalArray() {
  int a[5];
  return std::begin(a); // TODO
}

struct ptr_wrapper {
  std::vector<int>::iterator member;
};

ptr_wrapper getPtrWrapper();

std::vector<int>::iterator returnPtrFromWrapper() {
  ptr_wrapper local = getPtrWrapper();
  return local.member;
}

std::vector<int>::iterator returnPtrFromWrapperThroughRef() {
  ptr_wrapper local = getPtrWrapper();
  ptr_wrapper &local2 = local;
  return local2.member;
}

std::vector<int>::iterator returnPtrFromWrapperThroughRef2() {
  ptr_wrapper local = getPtrWrapper();
  std::vector<int>::iterator &local2 = local.member;
  return local2;
}

void checkPtrMemberFromAggregate() {
  std::vector<int>::iterator local = getPtrWrapper().member; // OK.
}

std::vector<int>::iterator doNotInterferWithUnannotated() {
  Unannotated value;
  // Conservative choice for now. Probably not ok, but we do not warn.
  return std::begin(value);
}

std::vector<int>::iterator doNotInterferWithUnannotated2() {
  Unannotated value;
  return value;
}

std::vector<int>::iterator supportDerefAddrofChain(int a, std::vector<int>::iterator value) {
  switch (a)  {
    default:
      return value;
    case 1:
      return *&value;
    case 2:
      return *&*&value;
    case 3:
      return *&*&*&value;
  }
}

int &supportDerefAddrofChain2(int a, std::vector<int>::iterator value) {
  switch (a)  {
    default:
      return *value;
    case 1:
      return **&value;
    case 2:
      return **&*&value;
    case 3:
      return **&*&*&value;
  }
}

int *supportDerefAddrofChain3(int a, std::vector<int>::iterator value) {
  switch (a)  {
    default:
      return &*value;
    case 1:
      return &*&*value;
    case 2:
      return &*&**&value;
    case 3:
      return &*&**&*&value;
  }
}

MyIntPointer handleDerivedToBaseCast1(MySpecialIntPointer ptr) {
  return ptr;
}

MyIntPointer handleDerivedToBaseCast2(MyOwnerIntPointer ptr) {
  return ptr; // expected-warning {{address of stack memory associated with parameter 'ptr' returned}}
}

std::vector<int>::iterator noFalsePositiveWithVectorOfPointers() {
  std::vector<std::vector<int>::iterator> iters;
  return iters.at(0);
}

void testForBug49342()
{
  auto it = std::iter<char>{} - 2; // Used to be false positive.
}

namespace GH93386 {
// verify no duplicated diagnostics are emitted.
struct [[gsl::Pointer]] S {
  S(const std::vector<int>& abc [[clang::lifetimebound]]);
};

S test(std::vector<int> a) {
  return S(a);  // expected-warning {{address of stack memory associated with}} \
                // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

// FIXME: Detect this using the CFG-based lifetime analysis (global initialisation).
auto s = S(std::vector<int>()); // expected-warning {{temporary whose address is used as value of local variable}}

// Verify no regression on the follow case.
std::string_view test2(int i, std::optional<std::string_view> a) {
  if (i)
    return std::move(*a);
  return std::move(a.value());
}

struct Foo;
struct FooView {
  FooView(const Foo& foo [[clang::lifetimebound]]);
};
FooView test3(int i, std::optional<Foo> a) {
  // FIXME: Detect this using the CFG-based lifetime analysis.
  //        Origin tracking for non-pointers type retured from lifetimebound fn is missing.
  //        https://github.com/llvm/llvm-project/issues/163600
  if (i)
    return *a; // expected-warning {{address of stack memory}}
  return a.value(); // expected-warning {{address of stack memory}}
}
} // namespace GH93386

namespace GH100549 {
struct UrlAnalyzed {
  UrlAnalyzed(std::string_view url [[clang::lifetimebound]]);
};
std::string StrCat(std::string_view, std::string_view);
void test1() {
  // FIXME: Detect this using the CFG-based lifetime analysis.
  //        Origin tracking for non-pointers type retured from lifetimebound fn is missing.
  //        https://github.com/llvm/llvm-project/issues/163600
  UrlAnalyzed url(StrCat("abc", "bcd")); // expected-warning {{object backing the pointer will be destroyed}}
  use(url);
}

std::string_view ReturnStringView(std::string_view abc [[clang::lifetimebound]]);

void test() {
  std::string_view svjkk1 = ReturnStringView(StrCat("bar", "x")); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}} \
                                                                  // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(svjkk1);                                                    // cfg-note {{later used here}}
}
} // namespace GH100549

namespace GH108272 {
template <typename T>
struct [[gsl::Owner]] StatusOr {
  const T &value() [[clang::lifetimebound]];
  // TODO: Do this behind a macro and run tests without this dtor to verify trivial dtor cases.
  ~StatusOr();
};

template <typename V>
class Wrapper1 {
 public:
  operator V() const;
  V value;
};
std::string_view test1() {
  StatusOr<Wrapper1<std::string_view>> k;
  // Be conservative in this case, as there is not enough information available
  // to infer the lifetime relationship for the Wrapper1 type.
  std::string_view good = StatusOr<Wrapper1<std::string_view>>().value();
  return k.value();
}

template <typename V>
class Wrapper2 {
 public:
  operator V() const [[clang::lifetimebound]];
  V value;
};
std::string_view test2() {
  StatusOr<Wrapper2<std::string_view>> k;
  // We expect dangling issues as the conversion operator is lifetimebound。
  std::string_view bad = StatusOr<Wrapper2<std::string_view>>().value(); // expected-warning {{temporary whose address is used as value of}}
  
  return k.value(); // expected-warning {{address of stack memory associated}} \
                    // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}
} // namespace GH108272

namespace GH100526 {
// FIXME: Detect this using the CFG-based lifetime analysis.
//        Container of pointers
//        https://github.com/llvm/llvm-project/issues/175025
void test() {
  std::vector<std::string_view> v1({std::string()}); // expected-warning {{object backing the pointer will be destroyed at the end}}
  use(v1);

  std::vector<std::string_view> v2({
    std::string(), // expected-warning {{object backing the pointer will be destroyed at the end}}
    std::string_view()
  });
  use(v2);

  std::vector<std::string_view> v3({
    std::string_view(),
    std::string()  // expected-warning {{object backing the pointer will be destroyed at the end}}
  });
  use(v3);

  std::optional<std::string_view> o1 = std::string(); // expected-warning {{object backing the pointer}}
  use(o1);

  std::string s;
  // This is a tricky use-after-free case, what it does:
  //   1. make_optional creates a temporary "optional<string>"" object
  //   2. the temporary object owns the underlying string which is copied from s.
  //   3. the t3 object holds the view to the underlying string of the temporary object.
  std::optional<std::string_view> o2 = std::make_optional(s); // expected-warning {{object backing the pointer}}
  std::optional<std::string_view> o3 = std::optional<std::string>(s); // expected-warning {{object backing the pointer}}
  std::optional<std::string_view> o4 = std::optional<std::string_view>(s);
  use(o2, o3, o4);

  // FIXME: should work for assignment cases
  v1 = {std::string()};
  o1 = std::string();
  use(o1, v1);

  // no warning on copying pointers.
  std::vector<std::string_view> n1 = {std::string_view()};
  std::optional<std::string_view> n2 = {std::string_view()};
  std::optional<std::string_view> n3 = std::string_view();
  std::optional<std::string_view> n4 = std::make_optional(std::string_view());
  const char* b = "";
  std::optional<std::string_view> n5 = std::make_optional(b);
  std::optional<std::string_view> n6 = std::make_optional("test");
  use(n1, n2, n3, n4, n5, n6);
}

std::vector<std::string_view> test2(int i) {
  std::vector<std::string_view> t;
  if (i)
    return t; // this is fine, no dangling
  return std::vector<std::string_view>(t.begin(), t.end());
}

class Foo {
  public:
   operator std::string_view() const { return ""; }
};
class [[gsl::Owner]] FooOwner {
  public:
   operator std::string_view() const { return ""; }
};
std::optional<Foo> GetFoo();
std::optional<FooOwner> GetFooOwner();

template <typename T>
struct [[gsl::Owner]] Container1 {
   Container1();
};
template <typename T>
struct [[gsl::Owner]] Container2 {
  template<typename U>
  Container2(const Container1<U>& C2);
};

std::optional<std::string_view> test3(int i) {
  std::string s;
  std::string_view sv;
  if (i)
   return s; // expected-warning {{address of stack memory associated}}
  return sv; // fine
  Container2<std::string_view> c1 = Container1<Foo>(); // no diagnostic as Foo is not an Owner.
  Container2<std::string_view> c2 = Container1<FooOwner>(); // expected-warning {{object backing the pointer will be destroyed}}
  return GetFoo(); // fine, we don't know Foo is owner or not, be conservative.
  return GetFooOwner(); // expected-warning {{returning address of local temporary object}}
}

std::optional<int*> test4(int a) {
  return std::make_optional(nullptr); // fine
}


template <typename T>
struct [[gsl::Owner]] StatusOr {
  const T &valueLB() const [[clang::lifetimebound]];
  const T &valueNoLB() const;
};

template<typename T>
struct [[gsl::Pointer]] Span {
  Span(const std::vector<T> &V);

  const int& getFieldLB() const [[clang::lifetimebound]];
  const int& getFieldNoLB() const;
};


/////// From Owner<Pointer> ///////

// Pointer from Owner<Pointer>
std::string_view test5() {
  // The Owner<Pointer> doesn't own the object which its inner pointer points to.
  std::string_view a = StatusOr<std::string_view>().valueLB(); // OK
  return StatusOr<std::string_view>().valueLB(); // OK

  // No dangling diagnostics on non-lifetimebound methods.
  std::string_view b = StatusOr<std::string_view>().valueNoLB();
  return StatusOr<std::string_view>().valueNoLB();
}

// Pointer<Pointer> from Owner<Pointer>
// Prevent regression GH108463
Span<int*> test6(std::vector<int*> v) {
  Span<int *> dangling = std::vector<int*>(); // expected-warning {{object backing the pointer}}
  dangling = std::vector<int*>(); // expected-warning {{object backing the pointer}}
  return v; // expected-warning {{address of stack memory}} \
            // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

/////// From Owner<Owner<Pointer>> ///////

// Pointer from Owner<Owner<Pointer>>
int* test7(StatusOr<StatusOr<int*>> aa) {
  // No dangling diagnostic on pointer.
  return aa.valueLB().valueLB(); // OK.
}

// Owner<Pointer> from Owner<Owner<Pointer>>
std::vector<int*> test8(StatusOr<std::vector<int*>> aa) {
  return aa.valueLB(); // OK, no pointer being construct on this case.
  return aa.valueNoLB();
}

// Pointer<Pointer> from Owner<Owner<Pointer>>
Span<int*> test9(StatusOr<std::vector<int*>> aa) {
  return aa.valueLB(); // expected-warning {{address of stack memory associated}} \
                       // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
  return aa.valueNoLB(); // OK.
}

/////// From Owner<Owner> ///////

// Pointer<Owner>> from Owner<Owner>
Span<std::string> test10(StatusOr<std::vector<std::string>> aa) {
  return aa.valueLB(); // expected-warning {{address of stack memory}} \
                       // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
  return aa.valueNoLB(); // OK.
}

/////// From Owner<Pointer<Owner>> ///////

// Pointer<Owner>> from Owner<Pointer<Owner>>
Span<std::string> test11(StatusOr<Span<std::string>> aa) {
  return aa.valueLB(); // OK
  return aa.valueNoLB(); // OK.
}

// Lifetimebound and gsl::Pointer.
const int& test12(Span<int> a) {
  return a.getFieldLB(); // expected-warning {{reference to stack memory associated}} \
                         // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
  return a.getFieldNoLB(); // OK.
}

void test13() {
  // FIXME: RHS is Owner<Pointer>, we skip this case to avoid false positives.
  std::optional<Span<int*>> abc = std::vector<int*>{};

  // FIXME: Detect this using the CFG-based lifetime analysis (container of pointer).
  std::optional<Span<int>> t = std::vector<int> {}; // expected-warning {{object backing the pointer will be destroyed}}
  use(t);
}

} // namespace GH100526

namespace std {
template <typename T>
class __set_iterator {};

template<typename T>
struct BB {
  typedef  __set_iterator<T> iterator;
};

template <typename T>
class set {
public:
  ~set();
  typedef typename BB<T>::iterator iterator;
  iterator begin() const;
};
} // namespace std
namespace GH118064{

void test() {
  auto y = std::set<int>{}.begin(); // expected-warning {{object backing the pointer}}
  use(y);
}
} // namespace GH118064

namespace LifetimeboundInterleave {

const std::string& Ref(const std::string& abc [[clang::lifetimebound]]);

std::string_view TakeSv(std::string_view abc [[clang::lifetimebound]]);
std::string_view TakeStrRef(const std::string& abc [[clang::lifetimebound]]);
std::string_view TakeStr(std::string abc [[clang::lifetimebound]]);

std::string_view test1_1() {
  std::string_view t1 = Ref(std::string()); // expected-warning {{object backing}} \
                                            // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(t1);                                  // cfg-note {{later used here}}
  t1 = Ref(std::string()); // expected-warning {{object backing}} \
                           // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(t1);                 // cfg-note {{later used here}}
  return Ref(std::string()); // expected-warning {{returning address}} \
                             // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

std::string_view test1_2() {
  std::string_view t2 = TakeSv(std::string()); // expected-warning {{object backing}} \
                                            // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(t2);                                  // cfg-note {{later used here}}
  t2 = TakeSv(std::string()); // expected-warning {{object backing}} \
                              // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(t2);                    // cfg-note {{later used here}}

  return TakeSv(std::string()); // expected-warning {{returning address}} \
                                // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

std::string_view test1_3() {
  std::string_view t3 = TakeStrRef(std::string()); // expected-warning {{temporary}} \
                                                   // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(t3);                                         // cfg-note {{later used here}}
  t3 = TakeStrRef(std::string()); // expected-warning {{object backing}} \
                                  // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(t3);                        // cfg-note {{later used here}}
  return TakeStrRef(std::string()); // expected-warning {{returning address}} \
                                    // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

std::string_view test1_4() {
  std::string_view t4 = TakeStr(std::string());
  t4 = TakeStr(std::string());
  use(t4);
  return TakeStr(std::string());
}

template <typename T>
struct Foo {
  const T& get() const [[clang::lifetimebound]];
  const T& getNoLB() const;
  // TODO: Do this behind a macro and run tests without this dtor to verify trivial dtor cases.
  ~Foo();
};
std::string_view test2_1(Foo<std::string> r1, Foo<std::string_view> r2) {
  std::string_view t1 = Foo<std::string>().get(); // expected-warning {{object backing}} \
                                                  // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(t1);                                        // cfg-note {{later used here}}
  t1 = Foo<std::string>().get(); // expected-warning {{object backing}} \
                                 // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  use(t1);                       // cfg-note {{later used here}}
  return r1.get(); // expected-warning {{address of stack}} \
                   // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}
std::string_view test2_2(Foo<std::string> r1, Foo<std::string_view> r2) {
  std::string_view t2 = Foo<std::string_view>().get();
  use(t2);
  t2 = Foo<std::string_view>().get();
  use(t2);
  return r2.get();
}
std::string_view test2_3(Foo<std::string> r1, Foo<std::string_view> r2) {
  // no warning on no-LB-annotated method.
  std::string_view t3 = Foo<std::string>().getNoLB();
  use(t3);
  t3 = Foo<std::string>().getNoLB();
  use(t3);
  return r1.getNoLB();
}

struct Bar {
  // TODO: Do this behind a macro and run tests without this dtor to verify trivial dtor cases.
  ~Bar();
};
struct [[gsl::Pointer]] Pointer {
  Pointer(const Bar & bar [[clang::lifetimebound]]);
};
Pointer test3(Bar bar) {
  // FIXME: Detect this using the CFG-based lifetime analysis (constructor of a pointer).
  //        https://github.com/llvm/llvm-project/issues/175898
  Pointer p = Pointer(Bar()); // expected-warning {{temporary}}
  use(p);
  p = Pointer(Bar()); // expected-warning {{object backing}}
  use(p);
  return bar; // expected-warning {{address of stack}}
}

template<typename T>
struct MySpan {
  MySpan(const std::vector<T>& v);
  ~MySpan();
  using iterator = std::iterator<T>;
  // FIXME: It is not possible to annotate accessor methods of non-owning view types.
  // Clang should provide another annotation to mark such functions as 'transparent'.
  iterator begin() const;
};
// FIXME: Same as above.
template <typename T>
typename MySpan<T>::iterator ReturnFirstIt(const MySpan<T>& v);

void test4() {
  std::vector<int> v{1};
  // MySpan<T> doesn't own any underlying T objects, the pointee object of
  // the MySpan iterator is still alive when the whole span is destroyed, thus
  // no diagnostic.
  const int& t1 = *MySpan<int>(v).begin();
  const int& t2 = *ReturnFirstIt(MySpan<int>(v));
  // Ideally, we would diagnose the following case, but due to implementation
  // constraints, we do not.
  const int& t4 = *MySpan<int>(std::vector<int>{}).begin();
  use(t1, t2, t4);

  auto it1 = MySpan<int>(v).begin();
  auto it2 = ReturnFirstIt(MySpan<int>(v));
  use(it1, it2);
}

} // namespace LifetimeboundInterleave

namespace range_based_for_loop_variables {
std::string_view test_view_loop_var(std::vector<std::string> strings) {
  for (std::string_view s : strings) {  // cfg-warning {{address of stack memory is returned later}} 
    return s; //cfg-note {{returned here}}
  }
  return "";
}

const char* test_view_loop_var_with_data(std::vector<std::string> strings) {
  for (std::string_view s : strings) {  // cfg-warning {{address of stack memory is returned later}} 
    return s.data(); //cfg-note {{returned here}}
  }
  return "";
}

std::string_view test_no_error_for_views(std::vector<std::string_view> views) {
  for (std::string_view s : views) {
    return s;
  }
  return "";
}

std::string_view test_string_ref_var(std::vector<std::string> strings) {
  for (const std::string& s : strings) {  // cfg-warning {{address of stack memory is returned later}} 
    return s; //cfg-note {{returned here}}
  }
  return "";
}

std::string_view test_opt_strings(std::optional<std::vector<std::string>> strings_or) {
  for (const std::string& s : *strings_or) {  // cfg-warning {{address of stack memory is returned later}} 
    return s; //cfg-note {{returned here}}
  }
  return "";
}
} // namespace range_based_for_loop_variables

namespace iterator_arrow {
std::string_view test() {
  std::vector<std::string> strings;
  return strings.begin()->data(); // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}
}

void operator_star_arrow_reference() {
  std::vector<std::string> v;
  const char* p = v.begin()->data();
  const char* q = (*v.begin()).data();
  const std::string& r = *v.begin();

  auto temporary = []() { return std::vector<std::string>{{"1"}}; };
  const char* x = temporary().begin()->data();    // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  const char* y = (*temporary().begin()).data();  // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}
  const std::string& z = (*temporary().begin());  // cfg-warning {{object whose reference is captured does not live long enough}} cfg-note {{destroyed here}}

  use(p, q, r, x, y, z); // cfg-note 3 {{later used here}}
}

void operator_star_arrow_of_iterators_false_positive_no_cfg_analysis() {
  std::vector<std::pair<int, std::string>> v;
  const char* p = v.begin()->second.data();
  const char* q = (*v.begin()).second.data();
  const std::string& r = (*v.begin()).second;

  // FIXME: Detect this using the CFG-based lifetime analysis.
  //        Detect dangling references to struct field.
  //        https://github.com/llvm/llvm-project/issues/176144
  auto temporary = []() { return std::vector<std::pair<int, std::string>>{{1, "1"}}; };
  const char* x = temporary().begin()->second.data();
  const char* y = (*temporary().begin()).second.data();
  const std::string& z = (*temporary().begin()).second;

  use(p, q, r, x, y, z);
}
} // namespace iterator_arrow

namespace GH120206 {
struct S {
  std::string_view s;
};

struct [[gsl::Owner]] Q1 {
  const S* get() const [[clang::lifetimebound]];
};
std::string_view test1(int c, std::string_view sv) {
  std::string_view k = c > 1 ? Q1().get()->s : sv;
  if (c == 1)
    return  c > 1 ? Q1().get()->s : sv;
  Q1 q;
  return c > 1 ? q.get()->s : sv;
}

struct Q2 {
  const S* get() const [[clang::lifetimebound]];
};
std::string_view test2(int c, std::string_view sv) {
  std::string_view k = c > 1 ? Q2().get()->s : sv;
  if (c == 1)
    return c > 1 ? Q2().get()->s : sv;
  Q2 q;
  return c > 1 ? q.get()->s : sv;
}

} // namespace GH120206

namespace GH120543 {
struct S {
  std::string_view sv;
  std::string s;
};
struct Q {
  const S* get() const [[clang::lifetimebound]];
};

std::string_view foo(std::string_view sv [[clang::lifetimebound]]);

// FIXME: Detect this using the CFG-based lifetime analysis.
//        Detect dangling references to struct field.
//        https://github.com/llvm/llvm-project/issues/176144
void test1() {
  std::string_view k1 = S().sv; // OK
  std::string_view k2 = S().s; // expected-warning {{object backing the pointer will}}

  std::string_view k3 = Q().get()->sv; // OK
  std::string_view k4  = Q().get()->s; // expected-warning {{object backing the pointer will}}

  std::string_view lb1 = foo(S().s); // expected-warning {{object backing the pointer will}}
  std::string_view lb2 = foo(Q().get()->s); // expected-warning {{object backing the pointer will}}

  use(k1, k2, k3, k4, lb1, lb2);
}

struct Bar {};
struct Foo {
  std::vector<Bar> v;
};
Foo getFoo();
void test2() {
  const Foo& foo = getFoo();
  const Bar& bar = foo.v.back(); // OK
}

struct Foo2 {
   std::unique_ptr<Bar> bar;
};

struct Test {
  Test(Foo2 foo) : bar(foo.bar.get()), // OK
      storage(std::move(foo.bar)) {};

  Bar* bar;
  std::unique_ptr<Bar> storage;
};

} // namespace GH120543

namespace GH127195 {
template <typename T>
struct StatusOr {
  T* operator->() [[clang::lifetimebound]];
  T* value() [[clang::lifetimebound]];
};

const char* foo() {
  StatusOr<std::string> s;
  return s->data(); // expected-warning {{address of stack memory associated with local variable}} \
                    // cfg-warning {{address of stack memory is returned later}} cfg-note {{returned here}}

  StatusOr<std::string_view> s2;
  return s2->data();

  StatusOr<StatusOr<std::string_view>> s3;
  return s3.value()->value()->data();

  // FIXME: nested cases are not supported now.
  StatusOr<StatusOr<std::string>> s4;
  return s4.value()->value()->data();
}

} // namespace GH127195

// Lifetimebound on definition vs declaration on implicit this param.
namespace GH175391 {
// Version A: Attribute on declaration only
class StringA {
public:
    const char* data() const [[clang::lifetimebound]];  // Declaration with attribute
private:
    char buffer[32] = "hello";
};
inline const char* StringA::data() const {  // Definition WITHOUT attribute
    return buffer;
}

// Version B: Attribute on definition only
class StringB {
public:
    const char* data() const;  // No attribute
private:
    char buffer[32] = "hello";
};
inline const char* StringB::data() const [[clang::lifetimebound]] {
    return buffer;
}

// Version C: Attribute on BOTH declaration and definition
class StringC {
public:
    const char* data() const [[clang::lifetimebound]];
private:
    char buffer[32] = "hello";
};
inline const char* StringC::data() const [[clang::lifetimebound]] {
    return buffer;
}

// TEMPLATED VERSIONS

// Template Version A: Attribute on declaration only
template<typename T>
class StringTemplateA {
public:
    const T* data() const [[clang::lifetimebound]];  // Declaration with attribute
private:
    T buffer[32];
};
template<typename T>
inline const T* StringTemplateA<T>::data() const {  // Definition WITHOUT attribute
    return buffer;
}

// Template Version B: Attribute on definition only
template<typename T>
class StringTemplateB {
public:
    const T* data() const;  // No attribute
private:
    T buffer[32];
};
template<typename T>
inline const T* StringTemplateB<T>::data() const [[clang::lifetimebound]] {
    return buffer;
}

// Template Version C: Attribute on BOTH declaration and definition
template<typename T>
class StringTemplateC {
public:
    const T* data() const [[clang::lifetimebound]];
private:
    T buffer[32];
};
template<typename T>
inline const T* StringTemplateC<T>::data() const [[clang::lifetimebound]] {
    return buffer;
}

// TEMPLATE SPECIALIZATION VERSIONS

// Template predeclarations for specializations
template<typename T> class StringTemplateSpecA;
template<typename T> class StringTemplateSpecB;
template<typename T> class StringTemplateSpecC;

// Template Specialization Version A: Attribute on declaration only - <char> specialization
template<>
class StringTemplateSpecA<char> {
public:
    const char* data() const [[clang::lifetimebound]];  // Declaration with attribute
private:
    char buffer[32] = "hello";
};
inline const char* StringTemplateSpecA<char>::data() const {  // Definition WITHOUT attribute
    return buffer;
}

// Template Specialization Version B: Attribute on definition only - <char> specialization
template<>
class StringTemplateSpecB<char> {
public:
    const char* data() const;  // No attribute
private:
    char buffer[32] = "hello";
};
inline const char* StringTemplateSpecB<char>::data() const [[clang::lifetimebound]] {
    return buffer;
}

// Template Specialization Version C: Attribute on BOTH declaration and definition - <char> specialization
template<>
class StringTemplateSpecC<char> {
public:
    const char* data() const [[clang::lifetimebound]];
private:
    char buffer[32] = "hello";
};
inline const char* StringTemplateSpecC<char>::data() const [[clang::lifetimebound]] {
    return buffer;
}

void test() {
    // Non-templated tests
    const auto ptrA = StringA().data();  // Declaration-only attribute  // expected-warning {{temporary whose address is used}}
    const auto ptrB = StringB().data();  // Definition-only attribute   // expected-warning {{temporary whose address is used}}
    const auto ptrC = StringC().data();  // Both have attribute         // expected-warning {{temporary whose address is used}}

    // Templated tests (generic templates)
    const auto ptrTA = StringTemplateA<char>().data();  // Declaration-only attribute // expected-warning {{temporary whose address is used}}
    // FIXME: Definition is not instantiated until the end of TU. The attribute is not merged when this call is processed.
    const auto ptrTB = StringTemplateB<char>().data();  // Definition-only attribute
    const auto ptrTC = StringTemplateC<char>().data();  // Both have attribute        // expected-warning {{temporary whose address is used}}

    // Template specialization tests
    const auto ptrTSA = StringTemplateSpecA<char>().data();  // Declaration-only attribute  // expected-warning {{temporary whose address is used}}
    const auto ptrTSB = StringTemplateSpecB<char>().data();  // Definition-only attribute   // expected-warning {{temporary whose address is used}}
    const auto ptrTSC = StringTemplateSpecC<char>().data();  // Both have attribute         // expected-warning {{temporary whose address is used}}
}
} // namespace GH175391

// RUN: %clang_cc1 --std=c++20 -fsyntax-only -Wdangling -Wdangling-field -Wreturn-stack-address -verify %s
struct [[gsl::Owner(int)]] MyIntOwner {
  MyIntOwner();
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
  return &j; // TODO
}

MyIntPointer returningLocalPointer() {
  MyIntPointer localPointer;
  return localPointer; // ok
}

MyIntPointer daglingGslPtrFromLocalOwner() {
  MyIntOwner localOwner;
  return localOwner; // expected-warning {{address of stack memory associated with local variable 'localOwner' returned}}
}

MyLongPointerFromConversion daglingGslPtrFromLocalOwnerConv() {
  MyLongOwnerWithConversion localOwner;
  return localOwner; // expected-warning {{address of stack memory associated with local variable 'localOwner' returned}}
}

MyIntPointer danglingGslPtrFromTemporary() {
  return MyIntOwner{}; // expected-warning {{returning address of local temporary object}}
}

MyIntOwner makeTempOwner();

MyIntPointer danglingGslPtrFromTemporary2() {
  return makeTempOwner(); // expected-warning {{returning address of local temporary object}}
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
  MyIntPointer p = MyIntOwner{}; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  MyIntPointer pp = p = MyIntOwner{}; // expected-warning {{object backing the pointer p will be}}
  p = MyIntOwner{}; // expected-warning {{object backing the pointer p }}
  pp = p; // no warning
  global = MyIntOwner{}; // expected-warning {{object backing the pointer global }}
  MyLongPointerFromConversion p2 = MyLongOwnerWithConversion{}; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  p2 = MyLongOwnerWithConversion{}; // expected-warning {{object backing the pointer p2 }}
  global2 = MyLongOwnerWithConversion{}; // expected-warning {{object backing the pointer global2 }}
}

namespace __gnu_cxx {
template <typename T>
struct basic_iterator {
  basic_iterator operator++();
  T& operator*() const;
  T* operator->() const;
};

template<typename T>
bool operator!=(basic_iterator<T>, basic_iterator<T>);
}

namespace std {
template<typename T> struct remove_reference       { typedef T type; };
template<typename T> struct remove_reference<T &>  { typedef T type; };
template<typename T> struct remove_reference<T &&> { typedef T type; };

template<typename T>
typename remove_reference<T>::type &&move(T &&t) noexcept;

template <typename C>
auto data(const C &c) -> decltype(c.data());

template <typename C>
auto begin(C &c) -> decltype(c.begin());

template<typename T, int N>
T *begin(T (&array)[N]);

using size_t = decltype(sizeof(0));

template<typename T>
struct initializer_list {
  const T* ptr; size_t sz;
};
template<typename T> class allocator {};
template <typename T, typename Alloc = allocator<T>>
struct vector {
  typedef __gnu_cxx::basic_iterator<T> iterator;
  iterator begin();
  iterator end();
  const T *data() const;
  vector();
  vector(initializer_list<T> __l,
         const Alloc& alloc = Alloc());

  template<typename InputIterator>
	vector(InputIterator first, InputIterator __last);

  T &at(int n);
};

template<typename T>
struct basic_string_view {
  basic_string_view();
  basic_string_view(const T *);
  const T *begin() const;
};
using string_view = basic_string_view<char>;

template<class _Mystr> struct iter {
    iter& operator-=(int);

    iter operator-(int _Off) const {
        iter _Tmp = *this;
        return _Tmp -= _Off;
    }
};

template<typename T>
struct basic_string {
  basic_string();
  basic_string(const T *);
  const T *c_str() const;
  operator basic_string_view<T> () const;
  using const_iterator = iter<T>;
};
using string = basic_string<char>;

template<typename T>
struct unique_ptr {
  T &operator*();
  T *get() const;
};

template<typename T>
struct optional {
  optional();
  optional(const T&);

  template<typename U = T>
  optional(U&& t);

  template<typename U>
  optional(optional<U>&& __t);

  T &operator*() &;
  T &&operator*() &&;
  T &value() &;
  T &&value() &&;
};
template<typename T>
optional<__decay(T)> make_optional(T&&);


template<typename T>
struct stack {
  T &top();
};

struct any {};

template<typename T>
T any_cast(const any& operand);

template<typename T>
struct reference_wrapper {
  template<typename U>
  reference_wrapper(U &&);
};

template<typename T>
reference_wrapper<T> ref(T& t) noexcept;

struct false_type {
    static constexpr bool value = false;
    constexpr operator bool() const noexcept { return value; }
};
struct true_type {
    static constexpr bool value = true;
    constexpr operator bool() const noexcept { return value; }
};

template<class T> struct is_pointer : false_type {};
template<class T> struct is_pointer<T*> : true_type {};
template<class T> struct is_pointer<T* const> : true_type {};
}

struct Unannotated {
  typedef std::vector<int>::iterator iterator;
  iterator begin();
  operator iterator() const;
};

void modelIterators() {
  std::vector<int>::iterator it = std::vector<int>().begin(); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  (void)it;
}

std::vector<int>::iterator modelIteratorReturn() {
  return std::vector<int>().begin(); // expected-warning {{returning address of local temporary object}}
}

const int *modelFreeFunctions() {
  return std::data(std::vector<int>()); // expected-warning {{returning address of local temporary object}}
}

int &modelAnyCast() {
  return std::any_cast<int&>(std::any{}); // expected-warning {{returning reference to local temporary object}}
}

int modelAnyCast2() {
  return std::any_cast<int>(std::any{}); // ok
}

int modelAnyCast3() {
  return std::any_cast<int&>(std::any{}); // ok
}

const char *danglingRawPtrFromLocal() {
  std::basic_string<char> s;
  return s.c_str(); // expected-warning {{address of stack memory associated with local variable 's' returned}}
}

int &danglingRawPtrFromLocal2() {
  std::optional<int> o;
  return o.value(); // expected-warning {{reference to stack memory associated with local variable 'o' returned}}
}

int &danglingRawPtrFromLocal3() {
  std::optional<int> o;
  return *o; // expected-warning {{reference to stack memory associated with local variable 'o' returned}}
}

// GH100384
std::string_view containerWithAnnotatedElements() {
  std::string_view c1 = std::vector<std::string>().at(0); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  c1 = std::vector<std::string>().at(0); // expected-warning {{object backing the pointer}}

  // no warning on constructing from gsl-pointer
  std::string_view c2 = std::vector<std::string_view>().at(0);

  std::vector<std::string> local;
  return local.at(0); // expected-warning {{address of stack memory associated with local variable}}
}

std::string_view localUniquePtr(int i) {
  std::unique_ptr<std::string> c1;
  if (i)
    return *c1; // expected-warning {{address of stack memory associated with local variable}}
  std::unique_ptr<std::string_view> c2;
  return *c2; // expect no-warning.
}

std::string_view localOptional(int i) {
  std::optional<std::string> o;
  if (i)
    return o.value(); // expected-warning {{address of stack memory associated with local variable}}
  std::optional<std::string_view> abc;
  return abc.value(); // expect no warning
}

const char *danglingRawPtrFromTemp() {
  return std::basic_string<char>().c_str(); // expected-warning {{returning address of local temporary object}}
}

std::unique_ptr<int> getUniquePtr();

int *danglingUniquePtrFromTemp() {
  return getUniquePtr().get(); // expected-warning {{returning address of local temporary object}}
}

int *danglingUniquePtrFromTemp2() {
  return std::unique_ptr<int>().get(); // expected-warning {{returning address of local temporary object}}
}

void danglingReferenceFromTempOwner() {
  int &&r = *std::optional<int>();          // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  int &&r2 = *std::optional<int>(5);        // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  int &&r3 = std::optional<int>(5).value(); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  int &r4 = std::vector<int>().at(3);       // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
}

std::vector<int> getTempVec();
std::optional<std::vector<int>> getTempOptVec();

void testLoops() {
  for (auto i : getTempVec()) // ok
    ;
  for (auto i : *getTempOptVec()) // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
    ;
}

int &usedToBeFalsePositive(std::vector<int> &v) {
  std::vector<int>::iterator it = v.begin();
  int& value = *it;
  return value; // ok
}

int &doNotFollowReferencesForLocalOwner() {
  std::unique_ptr<int> localOwner;
  int &p = *localOwner.get();
  // In real world code localOwner is usually moved here.
  return p; // ok
}

const char *trackThroughMultiplePointer() {
  return std::basic_string_view<char>(std::basic_string<char>()).begin(); // expected-warning {{returning address of local temporary object}}
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
    std::basic_string_view<char> v = cond ? def : ""; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
}

std::string operator+(std::string_view s1, std::string_view s2);
void danglingStringviewAssignment(std::string_view a1, std::string_view a2) {
  a1 = std::string(); // expected-warning {{object backing}}
  a2 = a1 + a1; // expected-warning {{object backing}}
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
  return S(a);  // expected-warning {{address of stack memory associated with}}
}

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
  UrlAnalyzed url(StrCat("abc", "bcd")); // expected-warning {{object backing the pointer will be destroyed}}
}

std::string_view ReturnStringView(std::string_view abc [[clang::lifetimebound]]);

void test() {
  std::string_view svjkk1 = ReturnStringView(StrCat("bar", "x")); // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
}
} // namespace GH100549

namespace GH108272 {
template <typename T>
struct [[gsl::Owner]] StatusOr {
  const T &value() [[clang::lifetimebound]];
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
  return k.value(); // expected-warning {{address of stack memory associated}}
}
} // namespace GH108272

namespace GH100526 {
void test() {
  std::vector<std::string_view> v1({std::string()}); // expected-warning {{object backing the pointer will be destroyed at the end}}
  std::vector<std::string_view> v2({
    std::string(), // expected-warning {{object backing the pointer will be destroyed at the end}}
    std::string_view()
  });
  std::vector<std::string_view> v3({
    std::string_view(),
    std::string()  // expected-warning {{object backing the pointer will be destroyed at the end}}
  });

  std::optional<std::string_view> o1 = std::string(); // expected-warning {{object backing the pointer}}

  std::string s;
  // This is a tricky use-after-free case, what it does:
  //   1. make_optional creates a temporary "optional<string>"" object
  //   2. the temporary object owns the underlying string which is copied from s.
  //   3. the t3 object holds the view to the underlying string of the temporary object.
  std::optional<std::string_view> o2 = std::make_optional(s); // expected-warning {{object backing the pointer}}
  std::optional<std::string_view> o3 = std::optional<std::string>(s); // expected-warning {{object backing the pointer}}
  std::optional<std::string_view> o4 = std::optional<std::string_view>(s);

  // FIXME: should work for assignment cases
  v1 = {std::string()};
  o1 = std::string();

  // no warning on copying pointers.
  std::vector<std::string_view> n1 = {std::string_view()};
  std::optional<std::string_view> n2 = {std::string_view()};
  std::optional<std::string_view> n3 = std::string_view();
  std::optional<std::string_view> n4 = std::make_optional(std::string_view());
  const char* b = "";
  std::optional<std::string_view> n5 = std::make_optional(b);
  std::optional<std::string_view> n6 = std::make_optional("test");
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
  std::string_view a = StatusOr<std::string_view>().valueLB(); // expected-warning {{object backing the pointer will be dest}}
  return StatusOr<std::string_view>().valueLB(); // expected-warning {{returning address of local temporary}}

  // No dangling diagnostics on non-lifetimebound methods.
  std::string_view b = StatusOr<std::string_view>().valueNoLB();
  return StatusOr<std::string_view>().valueNoLB();
}

// Pointer<Pointer> from Owner<Pointer>
// Prevent regression GH108463
Span<int*> test6(std::vector<int*> v) {
  Span<int *> dangling = std::vector<int*>(); // expected-warning {{object backing the pointer}}
  dangling = std::vector<int*>(); // expected-warning {{object backing the pointer}}
  return v; // expected-warning {{address of stack memory}}
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
  return aa.valueLB(); // expected-warning {{address of stack memory associated}}
  return aa.valueNoLB(); // OK.
}

/////// From Owner<Owner> ///////

// Pointer<Owner>> from Owner<Owner>
Span<std::string> test10(StatusOr<std::vector<std::string>> aa) {
  return aa.valueLB(); // expected-warning {{address of stack memory}}
  return aa.valueNoLB(); // OK.
}

/////// From Owner<Pointer<Owner>> ///////

// Pointer<Owner>> from Owner<Pointer<Owner>>
Span<std::string> test11(StatusOr<Span<std::string>> aa) {
  return aa.valueLB(); // expected-warning {{address of stack memory}}
  return aa.valueNoLB(); // OK.
}

// Lifetimebound and gsl::Pointer.
const int& test12(Span<int> a) {
  return a.getFieldLB(); // expected-warning {{reference to stack memory associated}}
  return a.getFieldNoLB(); // OK.
}

void test13() {
  // FIXME: RHS is Owner<Pointer>, we skip this case to avoid false positives.
  std::optional<Span<int*>> abc = std::vector<int*>{};

  std::optional<Span<int>> t = std::vector<int> {}; // expected-warning {{object backing the pointer will be destroyed}}
}

} // namespace GH100526

namespace lifetime_capture_by {
struct X {
  const int *x;
  void captureInt(const int& x [[clang::lifetime_capture_by(this)]]) { this->x = &x; }
  void captureSV(std::string_view sv [[clang::lifetime_capture_by(this)]]);
};
///////////////////////////
// Detect dangling cases.
///////////////////////////
void captureInt(const int& i [[clang::lifetime_capture_by(x)]], X& x);
void captureRValInt(int&& i [[clang::lifetime_capture_by(x)]], X& x);
void noCaptureInt(int i [[clang::lifetime_capture_by(x)]], X& x);

std::string_view substr(const std::string& s [[clang::lifetimebound]]);
std::string_view strcopy(const std::string& s);

void captureSV(std::string_view s [[clang::lifetime_capture_by(x)]], X& x);
void captureRValSV(std::string_view&& sv [[clang::lifetime_capture_by(x)]], X& x);
void noCaptureSV(std::string_view sv, X& x);
void captureS(const std::string& s [[clang::lifetime_capture_by(x)]], X& x);
void captureRValS(std::string&& s [[clang::lifetime_capture_by(x)]], X& x);

const std::string& getLB(const std::string& s[[clang::lifetimebound]]);
const std::string& getLB(std::string_view sv[[clang::lifetimebound]]);
const std::string* getPointerLB(const std::string& s[[clang::lifetimebound]]);
const std::string* getPointerNoLB(const std::string& s);

void capturePointer(const std::string* sp [[clang::lifetime_capture_by(x)]], X& x);

struct ThisIsCaptured {
  void capture(X& x) [[clang::lifetime_capture_by(x)]];
  void bar(X& x) [[clang::lifetime_capture_by(abcd)]]; // expected-error {{'lifetime_capture_by' attribute argument 'abcd' is not a known function parameter}}
  void baz(X& x) [[clang::lifetime_capture_by(this)]]; // expected-error {{'lifetime_capture_by' argument references itself}}
};

void captureByGlobal(std::string_view s [[clang::lifetime_capture_by(global)]]);
void captureByUnknown(std::string_view s [[clang::lifetime_capture_by(unknown)]]);

void use() {
  std::string_view local_sv;
  std::string local_s;
  X x;
  // Capture an 'int'.
  int local;
  captureInt(1, // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
            x);
  captureRValInt(1, x); // expected-warning {{object whose reference is captured by 'x'}}
  captureInt(local, x);
  noCaptureInt(1, x);
  noCaptureInt(local, x);

  // Capture using std::string_view.
  captureSV(local_sv, x);
  captureSV(std::string(), // expected-warning {{object whose reference is captured by 'x'}}
            x);
  captureSV(substr(
      std::string() // expected-warning {{object whose reference is captured by 'x'}}
      ), x);
  captureSV(substr(local_s), x);
  captureSV(strcopy(std::string()), x);
  captureRValSV(std::move(local_sv), x);
  captureRValSV(std::string(), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureRValSV(std::string_view{"abcd"}, x);
  captureRValSV(substr(local_s), x);
  captureRValSV(substr(std::string()), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureRValSV(strcopy(std::string()), x);
  noCaptureSV(local_sv, x);
  noCaptureSV(std::string(), x);
  noCaptureSV(substr(std::string()), x);

  // Capture using std::string.
  captureS(std::string(), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureS(local_s, x);
  captureRValS(std::move(local_s), x);
  captureRValS(std::string(), x); // expected-warning {{object whose reference is captured by 'x'}}

  // Capture with lifetimebound.
  captureSV(getLB(std::string()), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureSV(getLB(substr(std::string())), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureSV(getLB(getLB(
    std::string()  // expected-warning {{object whose reference is captured by 'x'}}
    )), x);
  capturePointer(getPointerLB(std::string()), x); // expected-warning {{object whose reference is captured by 'x'}}
  capturePointer(getPointerLB(*getPointerLB(
    std::string()  // expected-warning {{object whose reference is captured by 'x'}}
    )), x);
  capturePointer(getPointerNoLB(std::string()), x);

  // Member functions.
  x.captureInt(1); // expected-warning {{object whose reference is captured by 'x'}}
  x.captureSV(std::string()); // expected-warning {{object whose reference is captured by 'x'}}
  x.captureSV(substr(std::string())); // expected-warning {{object whose reference is captured by 'x'}}
  x.captureSV(strcopy(std::string()));

  // 'this' is captured.
  ThisIsCaptured{}.capture(x); // expected-warning {{object whose reference is captured by 'x'}}
  ThisIsCaptured TIS;
  TIS.capture(x);

  // capture by global.
  captureByGlobal(std::string()); // expected-warning {{object whose reference is captured will be destroyed at the end of the full-expression}}
  captureByGlobal(substr(std::string())); // expected-warning {{captured}}
  captureByGlobal(local_s);
  captureByGlobal(local_sv);

  // // capture by unknown.
  captureByGlobal(std::string()); // expected-warning {{object whose reference is captured will be destroyed at the end of the full-expression}}
  captureByGlobal(substr(std::string())); // expected-warning {{captured}}
  captureByGlobal(local_s);
  captureByGlobal(local_sv);
}

template<typename T> struct IsPointerLikeTypeImpl : std::false_type {};
template<> struct IsPointerLikeTypeImpl<std::string_view> : std::true_type {};
template<typename T> concept IsPointerLikeType = std::is_pointer<T>::value || IsPointerLikeTypeImpl<T>::value;

// Templated containers having no distinction between pointer-like and other element type.
template<class T>
struct MySet {
  void insert(T&& t [[clang::lifetime_capture_by(this)]]);
  void insert(const T& t [[clang::lifetime_capture_by(this)]]);
};
void user_defined_containers() {
  MySet<int> set_of_int;
  set_of_int.insert(1); // expected-warning {{object whose reference is captured by 'set_of_int' will be destroyed}}
  MySet<std::string_view> set_of_sv;
  set_of_sv.insert(std::string());  // expected-warning {{object whose reference is captured by 'set_of_sv' will be destroyed}}
}

// Templated containers having **which distinguishes** between pointer-like and other element type.
template<class T>
struct MyVector {
  void push_back(T&& t [[clang::lifetime_capture_by(this)]]) requires IsPointerLikeType<T>;
  void push_back(const T& t [[clang::lifetime_capture_by(this)]]) requires IsPointerLikeType<T>;

  void push_back(T&& t) requires (!IsPointerLikeType<T>);
  void push_back(const T& t) requires (!IsPointerLikeType<T>);
};

// Container of pointers.
struct [[gsl::Pointer()]] MyStringView : public std::string_view {
  MyStringView();
  MyStringView(std::string_view&&);
  MyStringView(const MyStringView&);
  MyStringView(const std::string&);
};
template<> struct IsPointerLikeTypeImpl<MyStringView> : std::true_type {};

std::optional<std::string_view> getOptionalSV();
std::optional<std::string> getOptionalS();
std::optional<MyStringView> getOptionalMySV();
MyStringView getMySV();

class MyStringViewNotPointer : public std::string_view {};
std::optional<MyStringViewNotPointer> getOptionalMySVNotP();
MyStringViewNotPointer getMySVNotP();

void container_of_pointers() {
  std::string local;
  MyVector<std::string> vs;
  vs.push_back(std::string()); // Ok.
  
  MyVector<std::string_view> vsv;
  vsv.push_back(std::string()); // expected-warning {{object whose reference is captured by 'vsv'}}
  vsv.push_back(substr(std::string())); // expected-warning {{object whose reference is captured by 'vsv'}}
  
  MyVector<const std::string*> vp;
  vp.push_back(getPointerLB(std::string())); // expected-warning {{object whose reference is captured by 'vp'}}
  vp.push_back(getPointerLB(*getPointerLB(std::string()))); // expected-warning {{object whose reference is captured by 'vp'}}
  vp.push_back(getPointerLB(local));
  vp.push_back(getPointerNoLB(std::string()));
  
  // User-defined [[gsl::Pointer]]
  vsv.push_back(getMySV());
  vsv.push_back(getMySVNotP());

  // Vector of user defined gsl::Pointer.
  MyVector<MyStringView> vmysv;
  vmysv.push_back(getMySV());
  vmysv.push_back(MyStringView{});
  vmysv.push_back(std::string_view{});
  vmysv.push_back(std::string{}); // expected-warning {{object whose reference is captured by 'vmysv'}}
  vmysv.push_back(substr(std::string{})); // expected-warning {{object whose reference is captured by 'vmysv'}}
  vmysv.push_back(getLB(substr(std::string{}))); // expected-warning {{object whose reference is captured by 'vmysv'}}
  vmysv.push_back(strcopy(getLB(substr(std::string{}))));

  // With std::optional container.
  std::optional<std::string_view> optional;
  vsv.push_back(optional.value());
  vsv.push_back(getOptionalS().value()); // expected-warning {{object whose reference is captured by 'vsv'}}
  vsv.push_back(getOptionalSV().value());
  vsv.push_back(getOptionalMySV().value());

  // (maybe) FIXME: We may choose to diagnose the following case.
  // This happens because 'MyStringViewNotPointer' is not marked as a [[gsl::Pointer]] but is derived from one.
  vsv.push_back(getOptionalMySVNotP().value()); // expected-warning {{object whose reference is captured by 'vsv'}}
}
} // namespace lifetime_capture_by

// RUN: %clang_cc1 -fsyntax-only -Wdangling -Wdangling-field -Wreturn-stack-address -verify %s
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
  MyIntPointer p = Y{}.a; // expected-warning {{temporary whose address is used as value of local variable 'p' will be destroyed at the end of the full-expression}}
  (void)p;
}

struct DanglingGslPtrField {
  MyIntPointer p; // expected-note 2{{pointer member declared here}}
  MyLongPointerFromConversion p2; // expected-note {{pointer member declared here}}
  DanglingGslPtrField(int i) : p(&i) {} // expected-warning {{initializing pointer member 'p' with the stack address of parameter 'i'}}
  DanglingGslPtrField() : p2(MyLongOwnerWithConversion{}) {} // expected-warning {{initializing pointer member 'p2' to point to a temporary object whose lifetime is shorter than the lifetime of the constructed object}}
  DanglingGslPtrField(double) : p(MyIntOwner{}) {} // expected-warning {{initializing pointer member 'p' to point to a temporary object whose lifetime is shorter than the lifetime of the constructed object}}
};

MyIntPointer danglingGslPtrFromLocal() {
  int j;
  return &j; // expected-warning {{address of stack memory associated with local variable 'j' returned}}
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
  p = MyIntOwner{}; // TODO ?
  global = MyIntOwner{}; // TODO ?
  MyLongPointerFromConversion p2 = MyLongOwnerWithConversion{}; // expected-warning {{object backing the pointer will be destroyed at the end of the full-expression}}
  p2 = MyLongOwnerWithConversion{}; // TODO ?
  global2 = MyLongOwnerWithConversion{}; // TODO ?
}

namespace __gnu_cxx {
template <typename T>
struct basic_iterator {
  basic_iterator operator++();
  T& operator*() const;
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

template <typename T>
struct vector {
  typedef __gnu_cxx::basic_iterator<T> iterator;
  iterator begin();
  iterator end();
  const T *data() const;
  T &at(int n);
};

template<typename T>
struct basic_string_view {
  basic_string_view(const T *);
  const T *begin() const;
};

template<typename T>
struct basic_string {
  const T *c_str() const;
  operator basic_string_view<T> () const;
};


template<typename T>
struct unique_ptr {
  T &operator*();
  T *get() const;
};

template<typename T>
struct optional {
  optional();
  optional(const T&);
  T &operator*() &;
  T &&operator*() &&;
  T &value() &;
  T &&value() &&;
};

template<typename T>
struct stack {
  T &top();
};

struct any {};

template<typename T>
T any_cast(const any& operand);
}

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

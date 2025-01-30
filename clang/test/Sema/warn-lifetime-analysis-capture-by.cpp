// RUN: %clang_cc1 --std=c++20 -fsyntax-only -verify -Wdangling-capture %s

#include "Inputs/lifetime-analysis.h"

// ****************************************************************************
// Capture an integer
// ****************************************************************************
namespace capture_int {
struct X {} x;
void captureInt(const int &i [[clang::lifetime_capture_by(x)]], X &x);
void captureRValInt(int &&i [[clang::lifetime_capture_by(x)]], X &x);
void noCaptureInt(int i [[clang::lifetime_capture_by(x)]], X &x);

void use() {
  int local;
  captureInt(1, // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
            x);
  captureRValInt(1, x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureInt(local, x);
  noCaptureInt(1, x);
  noCaptureInt(local, x);
}
} // namespace capture_int

// ****************************************************************************
// Capture std::string (gsl owner types)
// ****************************************************************************
namespace capture_string {
struct X {} x;
void captureString(const std::string &s [[clang::lifetime_capture_by(x)]], X &x);
void captureRValString(std::string &&s [[clang::lifetime_capture_by(x)]], X &x);

void use() {
  std::string local_string;
  captureString(std::string(), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureString(local_string, x);
  captureRValString(std::move(local_string), x);
  captureRValString(std::string(), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
}
} // namespace capture_string

// ****************************************************************************
// Capture std::string_view (gsl pointer types)
// ****************************************************************************
namespace capture_string_view {
struct X {} x;
void captureStringView(std::string_view s [[clang::lifetime_capture_by(x)]], X &x);
void captureRValStringView(std::string_view &&sv [[clang::lifetime_capture_by(x)]], X &x);
void noCaptureStringView(std::string_view sv, X &x);

std::string_view getLifetimeBoundView(const std::string& s [[clang::lifetimebound]]);
std::string_view getNotLifetimeBoundView(const std::string& s);
const std::string& getLifetimeBoundString(const std::string &s [[clang::lifetimebound]]);
const std::string& getLifetimeBoundString(std::string_view sv [[clang::lifetimebound]]);

void use() {
  std::string_view local_string_view;
  std::string local_string;
  captureStringView(local_string_view, x);
  captureStringView(std::string(), // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
            x);

  captureStringView(getLifetimeBoundView(local_string), x);
  captureStringView(getNotLifetimeBoundView(std::string()), x);
  captureRValStringView(std::move(local_string_view), x);
  captureRValStringView(std::string(), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureRValStringView(std::string_view{"abcd"}, x);

  noCaptureStringView(local_string_view, x);
  noCaptureStringView(std::string(), x);

  // With lifetimebound functions.
  captureStringView(getLifetimeBoundView(
  std::string() // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  ), x);
  captureRValStringView(getLifetimeBoundView(local_string), x);
  captureRValStringView(getLifetimeBoundView(std::string()), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureRValStringView(getNotLifetimeBoundView(std::string()), x);
  noCaptureStringView(getLifetimeBoundView(std::string()), x);
  captureStringView(getLifetimeBoundString(std::string()), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureStringView(getLifetimeBoundString(getLifetimeBoundView(std::string())), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureStringView(getLifetimeBoundString(getLifetimeBoundString(
    std::string()  // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
    )), x);
}
} // namespace capture_string_view

// ****************************************************************************
// Capture pointer (eg: std::string*)
// ****************************************************************************
const std::string* getLifetimeBoundPointer(const std::string &s [[clang::lifetimebound]]);
const std::string* getNotLifetimeBoundPointer(const std::string &s);

namespace capture_pointer {
struct X {} x;
void capturePointer(const std::string* sp [[clang::lifetime_capture_by(x)]], X &x);
void use() {
  capturePointer(getLifetimeBoundPointer(std::string()), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  capturePointer(getLifetimeBoundPointer(*getLifetimeBoundPointer(
    std::string()  // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
    )), x);
  capturePointer(getNotLifetimeBoundPointer(std::string()), x);

}
} // namespace capture_pointer

// ****************************************************************************
// Arrays and initializer lists.
// ****************************************************************************
namespace init_lists {
struct X {} x;
void captureVector(const std::vector<int> &a [[clang::lifetime_capture_by(x)]], X &x);
void captureArray(int array [[clang::lifetime_capture_by(x)]] [2], X &x);
void captureInitList(std::initializer_list<int> abc [[clang::lifetime_capture_by(x)]], X &x);


std::initializer_list<int> getLifetimeBoundInitList(std::initializer_list<int> abc [[clang::lifetimebound]]);

void use() {
  captureVector({1, 2, 3}, x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureVector(std::vector<int>{}, x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  std::vector<int> local_vector;
  captureVector(local_vector, x);
  int local_array[2]; 
  captureArray(local_array, x);
  captureInitList({1, 2}, x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureInitList(getLifetimeBoundInitList({1, 2}), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
}
} // namespace init_lists

// ****************************************************************************
// Implicit object param 'this' is captured
// ****************************************************************************
namespace this_is_captured {
struct X {} x;
struct S {
  void capture(X &x) [[clang::lifetime_capture_by(x)]];
};
void use() {
  S{}.capture(x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  S s;
  s.capture(x);
}
} // namespace this_is_captured

namespace temporary_capturing_object {
struct S {
  void add(const int& x [[clang::lifetime_capture_by(this)]]);
};

void test() {
  // We still give an warning even the capturing object is a temoprary.
  // It is possible that the capturing object uses the captured object in its
  // destructor.
  S().add(1); // expected-warning {{object whose reference is captured}}
  S{}.add(1); // expected-warning {{object whose reference is captured}}
}
} // namespace ignore_temporary_class_object

// ****************************************************************************
// Capture by Global and Unknown.
// ****************************************************************************
namespace capture_by_global_unknown {
void captureByGlobal(std::string_view s [[clang::lifetime_capture_by(global)]]);
void captureByUnknown(std::string_view s [[clang::lifetime_capture_by(unknown)]]);

std::string_view getLifetimeBoundView(const std::string& s [[clang::lifetimebound]]);

void use() {  
  std::string_view local_string_view;
  std::string local_string;
  // capture by global.
  captureByGlobal(std::string()); // expected-warning {{object whose reference is captured will be destroyed at the end of the full-expression}}
  captureByGlobal(getLifetimeBoundView(std::string())); // expected-warning {{object whose reference is captured will be destroyed at the end of the full-expression}}
  captureByGlobal(local_string);
  captureByGlobal(local_string_view);

  // capture by unknown.
  captureByUnknown(std::string()); // expected-warning {{object whose reference is captured will be destroyed at the end of the full-expression}}
  captureByUnknown(getLifetimeBoundView(std::string())); // expected-warning {{object whose reference is captured will be destroyed at the end of the full-expression}}
  captureByUnknown(local_string);
  captureByUnknown(local_string_view);
}
} // namespace capture_by_global_unknown

// ****************************************************************************
// Member functions: Capture by 'this'
// ****************************************************************************
namespace capture_by_this {
struct S {
  void captureInt(const int& x [[clang::lifetime_capture_by(this)]]);
  void captureView(std::string_view sv [[clang::lifetime_capture_by(this)]]);
};
std::string_view getLifetimeBoundView(const std::string& s [[clang::lifetimebound]]);
std::string_view getNotLifetimeBoundView(const std::string& s);
const std::string& getLifetimeBoundString(const std::string &s [[clang::lifetimebound]]);

void use() {
  S s;
  s.captureInt(1); // expected-warning {{object whose reference is captured by 's' will be destroyed at the end of the full-expression}}
  s.captureView(std::string()); // expected-warning {{object whose reference is captured by 's' will be destroyed at the end of the full-expression}}
  s.captureView(getLifetimeBoundView(std::string())); // expected-warning {{object whose reference is captured by 's' will be destroyed at the end of the full-expression}}
  s.captureView(getLifetimeBoundString(std::string()));  // expected-warning {{object whose reference is captured by 's' will be destroyed at the end of the full-expression}}
  s.captureView(getNotLifetimeBoundView(std::string()));
}  
} // namespace capture_by_this

// ****************************************************************************
// Struct with field as a reference
// ****************************************************************************
namespace reference_field {
struct X {} x;
struct Foo {
  const int& b;
};
void captureField(Foo param [[clang::lifetime_capture_by(x)]], X &x);
void use() {
  captureField(Foo{
    1 // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  }, x);
  int local;
  captureField(Foo{local}, x);
}
} // namespace reference_field

// ****************************************************************************
// Capture default argument.
// ****************************************************************************
namespace default_arg {
struct X {} x;
void captureDefaultArg(X &x, std::string_view s [[clang::lifetime_capture_by(x)]] = std::string());

std::string_view getLifetimeBoundView(const std::string& s [[clang::lifetimebound]]);

void useCaptureDefaultArg() {
  X x;
  captureDefaultArg(x); // FIXME: Diagnose temporary default arg.
  captureDefaultArg(x, std::string("temp")); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  captureDefaultArg(x, getLifetimeBoundView(std::string())); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
  std::string local;
  captureDefaultArg(x, local);
}
} // namespace default_arg 

// ****************************************************************************
// Container: *No* distinction between pointer-like and other element type
// ****************************************************************************
namespace containers_no_distinction {
template<class T>
struct MySet {
  void insert(T&& t [[clang::lifetime_capture_by(this)]]);
  void insert(const T& t [[clang::lifetime_capture_by(this)]]);
};
void user_defined_containers() {
  MySet<int> set_of_int;
  set_of_int.insert(1); // expected-warning {{object whose reference is captured by 'set_of_int' will be destroyed at the end of the full-expression}}
  MySet<std::string_view> set_of_sv;
  set_of_sv.insert(std::string());  // expected-warning {{object whose reference is captured by 'set_of_sv' will be destroyed at the end of the full-expression}}
  set_of_sv.insert(std::string_view());
}
} // namespace containers_no_distinction

// ****************************************************************************
// Container: Different for pointer-like and other element type.
// ****************************************************************************
namespace conatiners_with_different {
template<typename T> struct IsPointerLikeTypeImpl : std::false_type {};
template<> struct IsPointerLikeTypeImpl<std::string_view> : std::true_type {};
template<typename T> concept IsPointerLikeType = std::is_pointer<T>::value || IsPointerLikeTypeImpl<T>::value;

template<class T> struct MyVector {
  void push_back(T&& t [[clang::lifetime_capture_by(this)]]) requires IsPointerLikeType<T>;
  void push_back(const T& t [[clang::lifetime_capture_by(this)]]) requires IsPointerLikeType<T>;

  void push_back(T&& t) requires (!IsPointerLikeType<T>);
  void push_back(const T& t) requires (!IsPointerLikeType<T>);
};

std::string_view getLifetimeBoundView(const std::string& s [[clang::lifetimebound]]);

void use_container() {
  std::string local;

  MyVector<std::string> vector_of_string;
  vector_of_string.push_back(std::string()); // Ok.
  
  MyVector<std::string_view> vector_of_view;
  vector_of_view.push_back(std::string()); // expected-warning {{object whose reference is captured by 'vector_of_view' will be destroyed at the end of the full-expression}}
  vector_of_view.push_back(getLifetimeBoundView(std::string())); // expected-warning {{object whose reference is captured by 'vector_of_view' will be destroyed at the end of the full-expression}}
  
  MyVector<const std::string*> vector_of_pointer;
  vector_of_pointer.push_back(getLifetimeBoundPointer(std::string())); // expected-warning {{object whose reference is captured by 'vector_of_pointer' will be destroyed at the end of the full-expression}}
  vector_of_pointer.push_back(getLifetimeBoundPointer(*getLifetimeBoundPointer(std::string()))); // expected-warning {{object whose reference is captured by 'vector_of_pointer' will be destroyed at the end of the full-expression}}
  vector_of_pointer.push_back(getLifetimeBoundPointer(local));
  vector_of_pointer.push_back(getNotLifetimeBoundPointer(std::string()));
}

// ****************************************************************************
// Container: For user defined view types
// ****************************************************************************
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

std::string_view getLifetimeBoundView(const std::string& s [[clang::lifetimebound]]);
std::string_view getNotLifetimeBoundView(const std::string& s);
const std::string& getLifetimeBoundString(const std::string &s [[clang::lifetimebound]]);
const std::string& getLifetimeBoundString(std::string_view sv [[clang::lifetimebound]]);

void use_my_view() {
  std::string local;
  MyVector<MyStringView> vector_of_my_view;
  vector_of_my_view.push_back(getMySV());
  vector_of_my_view.push_back(MyStringView{});
  vector_of_my_view.push_back(std::string_view{});
  vector_of_my_view.push_back(std::string{}); // expected-warning {{object whose reference is captured by 'vector_of_my_view' will be destroyed at the end of the full-expression}}
  vector_of_my_view.push_back(getLifetimeBoundView(std::string{})); // expected-warning {{object whose reference is captured by 'vector_of_my_view' will be destroyed at the end of the full-expression}}
  vector_of_my_view.push_back(getLifetimeBoundString(getLifetimeBoundView(std::string{}))); // expected-warning {{object whose reference is captured by 'vector_of_my_view' will be destroyed at the end of the full-expression}}
  vector_of_my_view.push_back(getNotLifetimeBoundView(getLifetimeBoundString(getLifetimeBoundView(std::string{}))));
  
  // Use with container of other view types.
  MyVector<std::string_view> vector_of_view;
  vector_of_view.push_back(getMySV());
  vector_of_view.push_back(getMySVNotP());
}

// ****************************************************************************
// Container: Use with std::optional<view> (owner<pointer> types)
// ****************************************************************************
void use_with_optional_view() {
  MyVector<std::string_view> vector_of_view;

  std::optional<std::string_view> optional_of_view;
  vector_of_view.push_back(optional_of_view.value());
  vector_of_view.push_back(getOptionalS().value()); // expected-warning {{object whose reference is captured by 'vector_of_view' will be destroyed at the end of the full-expression}}
  
  vector_of_view.push_back(getOptionalSV().value());
  vector_of_view.push_back(getOptionalMySV().value());
  vector_of_view.push_back(getOptionalMySVNotP().value());
}
} // namespace conatiners_with_different

// ****************************************************************************
// Capture 'temporary' views
// ****************************************************************************
namespace temporary_views {
void capture1(std::string_view s [[clang::lifetime_capture_by(x)]], std::vector<std::string_view>& x);

// Intended to capture the "string_view" itself
void capture2(const std::string_view& s [[clang::lifetime_capture_by(x)]], std::vector<std::string_view*>& x);
// Intended to capture the pointee of the "string_view"
void capture3(const std::string_view& s [[clang::lifetime_capture_by(x)]], std::vector<std::string_view>& x);

void use() {
  std::vector<std::string_view> x1;
  capture1(std::string(), x1); // expected-warning {{object whose reference is captured by 'x1' will be destroyed at the end of the full-expression}}
  capture1(std::string_view(), x1);

  std::vector<std::string_view*> x2;
  // Clang considers 'const std::string_view&' to refer to the owner
  // 'std::string' and not 'std::string_view'. Therefore no diagnostic here.
  capture2(std::string_view(), x2);
  capture2(std::string(), x2); // expected-warning {{object whose reference is captured by 'x2' will be destroyed at the end of the full-expression}}
  
  std::vector<std::string_view> x3;
  capture3(std::string_view(), x3);
  capture3(std::string(), x3); // expected-warning {{object whose reference is captured by 'x3' will be destroyed at the end of the full-expression}}
}
} // namespace temporary_views

// ****************************************************************************
// Inferring annotation for STL containers
// ****************************************************************************
namespace inferred_capture_by {
const std::string* getLifetimeBoundPointer(const std::string &s [[clang::lifetimebound]]);
const std::string* getNotLifetimeBoundPointer(const std::string &s);

std::string_view getLifetimeBoundView(const std::string& s [[clang::lifetimebound]]);
std::string_view getNotLifetimeBoundView(const std::string& s);
void use() {
  std::string local;
  std::vector<std::string_view> views;
  views.push_back(std::string()); // expected-warning {{object whose reference is captured by 'views' will be destroyed at the end of the full-expression}}
  views.insert(views.begin(), 
            std::string()); // expected-warning {{object whose reference is captured by 'views' will be destroyed at the end of the full-expression}}
  views.push_back(getLifetimeBoundView(std::string())); // expected-warning {{object whose reference is captured by 'views' will be destroyed at the end of the full-expression}}
  views.push_back(getNotLifetimeBoundView(std::string()));
  views.push_back(local);
  views.insert(views.end(), local);

  std::vector<std::string> strings;
  strings.push_back(std::string());
  strings.insert(strings.begin(), std::string());

  std::vector<const std::string*> pointers;
  pointers.push_back(getLifetimeBoundPointer(std::string()));
  pointers.push_back(&local);
}

namespace with_span {
// Templated view types.
template<typename T>
struct [[gsl::Pointer]] Span {
  Span(const std::vector<T> &V);
};

void use() {
  std::vector<Span<int>> spans;
  spans.push_back(std::vector<int>{1, 2, 3}); // expected-warning {{object whose reference is captured by 'spans' will be destroyed at the end of the full-expression}}
  std::vector<int> local;
  spans.push_back(local);
}
} // namespace with_span
} // namespace inferred_capture_by

namespace on_constructor {
struct T {
  T(const int& t [[clang::lifetime_capture_by(this)]]);
};
struct T2 {
  T2(const int& t [[clang::lifetime_capture_by(x)]], int& x);
};
struct T3 {
  T3(const T& t [[clang::lifetime_capture_by(this)]]);
};

int foo(const T& t);
int bar(const T& t[[clang::lifetimebound]]);

void test() {
  auto x = foo(T(1)); // OK. no diagnosic
  T(1); // OK. no diagnostic
  T t(1); // expected-warning {{temporary whose address is used}}
  auto y = bar(T(1)); // expected-warning {{temporary whose address is used}}
  T3 t3(T(1)); // expected-warning {{temporary whose address is used}}
    
  int a;
  T2(1, a); // expected-warning {{object whose reference is captured by}}
}
} // namespace on_constructor

namespace GH121391 {

struct Foo {};

template <typename T>
struct Container {
  const T& tt() [[clang::lifetimebound]];
};
template<typename T>
struct StatusOr {
   T* get() [[clang::lifetimebound]];
};
StatusOr<Container<const Foo*>> getContainer();

void test() {
  std::vector<const Foo*> vv;
  vv.push_back(getContainer().get()->tt()); // OK
}

} // namespace GH121391

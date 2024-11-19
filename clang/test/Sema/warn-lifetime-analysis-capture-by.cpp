// RUN: %clang_cc1 --std=c++20 -fsyntax-only -Wdangling -Wdangling-field -Wreturn-stack-address -verify %s

#include "Inputs/lifetime-analysis.h"

struct X {
  const int *x;
};
X x;

// ****************************************************************************
// Capture an integer
// ****************************************************************************
namespace capture_int {
void captureInt(const int &i [[clang::lifetime_capture_by(x)]], X &x);
void captureRValInt(int &&i [[clang::lifetime_capture_by(x)]], X &x);
void noCaptureInt(int i [[clang::lifetime_capture_by(x)]], X &x);

void use() {
  int local;
  captureInt(1, // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}}
            x);
  captureRValInt(1, x); // expected-warning {{object whose reference is captured by 'x'}}
  captureInt(local, x);
  noCaptureInt(1, x);
  noCaptureInt(local, x);
}
} // namespace capture_int

// ****************************************************************************
// Capture std::string (gsl owner types)
// ****************************************************************************
std::string_view getLifetimeBoundView(const std::string& s [[clang::lifetimebound]]);
std::string_view getNotLifetimeBoundView(const std::string& s);
const std::string& getLifetimeBoundString(const std::string &s [[clang::lifetimebound]]);
const std::string& getLifetimeBoundString(std::string_view sv [[clang::lifetimebound]]);

namespace capture_string {
void captureString(const std::string &s [[clang::lifetime_capture_by(x)]], X &x);
void captureRValString(std::string &&s [[clang::lifetime_capture_by(x)]], X &x);

void use() {
  std::string local_string;
  captureString(std::string(), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureString(local_string, x);
  captureRValString(std::move(local_string), x);
  captureRValString(std::string(), x); // expected-warning {{object whose reference is captured by 'x'}}
}
} // namespace capture_string

// ****************************************************************************
// Capture std::string_view (gsl pointer types)
// ****************************************************************************
namespace capture_string_view {
void captureStringView(std::string_view s [[clang::lifetime_capture_by(x)]], X &x);
void captureRValStringView(std::string_view &&sv [[clang::lifetime_capture_by(x)]], X &x);
void noCaptureStringView(std::string_view sv, X &x);

void use() {
  std::string_view local_string_view;
  std::string local_string;
  captureStringView(local_string_view, x);
  captureStringView(std::string(), // expected-warning {{object whose reference is captured by 'x'}}
            x);

  captureStringView(getLifetimeBoundView(local_string), x);
  captureStringView(getNotLifetimeBoundView(std::string()), x);
  captureRValStringView(std::move(local_string_view), x);
  captureRValStringView(std::string(), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureRValStringView(std::string_view{"abcd"}, x);

  noCaptureStringView(local_string_view, x);
  noCaptureStringView(std::string(), x);

  // With lifetimebound functions.
  captureStringView(getLifetimeBoundView(
  std::string() // expected-warning {{object whose reference is captured by 'x'}}
  ), x);
  captureRValStringView(getLifetimeBoundView(local_string), x);
  captureRValStringView(getLifetimeBoundView(std::string()), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureRValStringView(getNotLifetimeBoundView(std::string()), x);
  noCaptureStringView(getLifetimeBoundView(std::string()), x);
  captureStringView(getLifetimeBoundString(std::string()), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureStringView(getLifetimeBoundString(getLifetimeBoundView(std::string())), x); // expected-warning {{object whose reference is captured by 'x'}}
  captureStringView(getLifetimeBoundString(getLifetimeBoundString(
    std::string()  // expected-warning {{object whose reference is captured by 'x'}}
    )), x);
}
} // namespace capture_string_view

// ****************************************************************************
// Capture pointer (eg: std::string*)
// ****************************************************************************
const std::string* getLifetimeBoundPointer(const std::string &s [[clang::lifetimebound]]);
const std::string* getNotLifetimeBoundPointer(const std::string &s);

namespace capture_pointer {
void capturePointer(const std::string* sp [[clang::lifetime_capture_by(x)]], X &x);
void use() {
  capturePointer(getLifetimeBoundPointer(std::string()), x); // expected-warning {{object whose reference is captured by 'x'}}
  capturePointer(getLifetimeBoundPointer(*getLifetimeBoundPointer(
    std::string()  // expected-warning {{object whose reference is captured by 'x'}}
    )), x);
  capturePointer(getNotLifetimeBoundPointer(std::string()), x);

}
} // namespace capture_pointer

// ****************************************************************************
// Arrays and initializer lists.
// ****************************************************************************
namespace init_lists {
void captureVector(const std::vector<int> &a [[clang::lifetime_capture_by(x)]], X &x);
void captureArray(int array [[clang::lifetime_capture_by(x)]] [2], X &x);
void captureInitList(std::initializer_list<int> abc [[clang::lifetime_capture_by(x)]], X &x);

void use() {
  captureVector({1, 2, 3}, x); // expected-warning {{capture}}
  captureVector(std::vector<int>{}, x); // expected-warning {{capture}}
  std::vector<int> local_vector;
  captureVector(local_vector, x);
  int local_array[2]; 
  captureArray(local_array, x);
  captureInitList({1, 2}, x); // expected-warning {{capture}}
}
}

// ****************************************************************************
// Implicit object param 'this' is captured
// ****************************************************************************
namespace this_is_captured {
struct S {
  void capture(X &x) [[clang::lifetime_capture_by(x)]];
};
void use() {
  S{}.capture(x); // expected-warning {{object whose reference is captured by 'x'}}
  S s;
  s.capture(x);
}
} // namespace this_is_captured

// ****************************************************************************
// Capture by Global and Unknown.
// ****************************************************************************
namespace capture_by_global_unknown {
void captureByGlobal(std::string_view s [[clang::lifetime_capture_by(global)]]);
void captureByUnknown(std::string_view s [[clang::lifetime_capture_by(unknown)]]);

void use() {  
  std::string_view local_string_view;
  std::string local_string;
  // capture by global.
  captureByGlobal(std::string()); // expected-warning {{object whose reference is captured will be destroyed at the end of the full-expression}}
  captureByGlobal(getLifetimeBoundView(std::string())); // expected-warning {{captured}}
  captureByGlobal(local_string);
  captureByGlobal(local_string_view);

  // capture by unknown.
  captureByUnknown(std::string()); // expected-warning {{object whose reference is captured will be destroyed at the end of the full-expression}}
  captureByUnknown(getLifetimeBoundView(std::string())); // expected-warning {{captured}}
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
void use() {
  S s;
  s.captureInt(1); // expected-warning {{object whose reference is captured by 's'}}
  s.captureView(std::string()); // expected-warning {{captured}}
  s.captureView(getLifetimeBoundView(std::string())); // expected-warning {{captured}}
  s.captureView(getLifetimeBoundString(std::string()));  // expected-warning {{captured}}
  s.captureView(getNotLifetimeBoundView(std::string()));
}  
} // namespace capture_by_this

// ****************************************************************************
// Struct with field as a reference
// ****************************************************************************
namespace reference_field {
struct Foo {
  const int& b;
};
void captureField(Foo param [[clang::lifetime_capture_by(x)]], X &x);
void use() {
  captureField(Foo{
    1 // expected-warning {{capture}}
  }, x);
  int local;
  captureField(Foo{local}, x);
}
} // namespace reference_field

// ****************************************************************************
// Capture default argument.
// ****************************************************************************
namespace default_arg {
void captureDefaultArg(X &x, std::string_view s [[clang::lifetime_capture_by(x)]] = std::string());
void useCaptureDefaultArg() {
  X x;
  captureDefaultArg(x); // FIXME: Diagnose temporary default arg.
  captureDefaultArg(x, std::string("temp")); // expected-warning {{captured}}
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
  set_of_int.insert(1); // expected-warning {{object whose reference is captured by 'set_of_int' will be destroyed}}
  MySet<std::string_view> set_of_sv;
  set_of_sv.insert(std::string());  // expected-warning {{object whose reference is captured by 'set_of_sv' will be destroyed}}
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

void use_container() {
  std::string local;

  MyVector<std::string> vector_of_string;
  vector_of_string.push_back(std::string()); // Ok.
  
  MyVector<std::string_view> vector_of_view;
  vector_of_view.push_back(std::string()); // expected-warning {{object whose reference is captured by 'vector_of_view'}}
  vector_of_view.push_back(getLifetimeBoundView(std::string())); // expected-warning {{captured}}
  
  MyVector<const std::string*> vector_of_pointer;
  vector_of_pointer.push_back(getLifetimeBoundPointer(std::string())); // expected-warning {{captured}}
  vector_of_pointer.push_back(getLifetimeBoundPointer(*getLifetimeBoundPointer(std::string()))); // expected-warning {{captured}}
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

void use_my_view() {
  std::string local;
  MyVector<MyStringView> vector_of_my_view;
  vector_of_my_view.push_back(getMySV());
  vector_of_my_view.push_back(MyStringView{});
  vector_of_my_view.push_back(std::string_view{});
  vector_of_my_view.push_back(std::string{}); // expected-warning {{object whose reference is captured by 'vector_of_my_view'}}
  vector_of_my_view.push_back(getLifetimeBoundView(std::string{})); // expected-warning {{captured}}
  vector_of_my_view.push_back(getLifetimeBoundString(getLifetimeBoundView(std::string{}))); // expected-warning {{captured}}
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
  vector_of_view.push_back(getOptionalS().value()); // expected-warning {{captured}}
  
  // FIXME: Following 2 cases are false positives:
  vector_of_view.push_back(getOptionalSV().value()); // expected-warning {{captured}}
  vector_of_view.push_back(getOptionalMySV().value());  // expected-warning {{captured}}

  // (maybe) FIXME: We may choose to diagnose the following case.
  // This happens because 'MyStringViewNotPointer' is not marked as a [[gsl::Pointer]] but is derived from one.
  vector_of_view.push_back(getOptionalMySVNotP().value()); // expected-warning {{captured}}
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
  capture1(std::string(), x1); // expected-warning {{captured by 'x1'}}
  capture1(std::string_view(), x1);

  std::vector<std::string_view*> x2;
  // Clang considers 'const std::string_view&' to refer to the owner
  // 'std::string' and not 'std::string_view'. Therefore no diagnostic here.
  capture2(std::string_view(), x2);
  capture2(std::string(), x2); // expected-warning {{captured by 'x2'}}
  
  std::vector<std::string_view> x3;
  capture3(std::string_view(), x3);
  capture3(std::string(), x3); // expected-warning {{captured by 'x3'}}
}
} // namespace temporary_views

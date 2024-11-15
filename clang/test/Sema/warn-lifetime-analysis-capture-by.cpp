// RUN: %clang_cc1 --std=c++20 -fsyntax-only -Wdangling -Wdangling-field -Wreturn-stack-address -verify %s

#include "Inputs/lifetime-analysis.h"

struct X {
  const int *x;
  void captureInt(const int& x [[clang::lifetime_capture_by(this)]]) { this->x = &x; }
  void captureSV(std::string_view sv [[clang::lifetime_capture_by(this)]]);
};
///////////////////////////
// Detect dangling cases.
///////////////////////////
void captureInt(const int &i [[clang::lifetime_capture_by(x)]], X &x);
void captureRValInt(int &&i [[clang::lifetime_capture_by(x)]], X &x);
void noCaptureInt(int i [[clang::lifetime_capture_by(x)]], X &x);

std::string_view substr(const std::string& s [[clang::lifetimebound]]);
std::string_view strcopy(const std::string& s);

void captureSV(std::string_view s [[clang::lifetime_capture_by(x)]], X &x);
void captureRValSV(std::string_view &&sv [[clang::lifetime_capture_by(x)]], X &x);
void noCaptureSV(std::string_view sv, X &x);
void captureS(const std::string &s [[clang::lifetime_capture_by(x)]], X &x);
void captureRValS(std::string &&s [[clang::lifetime_capture_by(x)]], X &x);

const std::string& getLB(const std::string &s [[clang::lifetimebound]]);
const std::string& getLB(std::string_view sv [[clang::lifetimebound]]);
const std::string* getPointerLB(const std::string &s [[clang::lifetimebound]]);
const std::string* getPointerNoLB(const std::string &s);

void capturePointer(const std::string* sp [[clang::lifetime_capture_by(x)]], X &x);

struct ThisIsCaptured {
  void capture(X &x) [[clang::lifetime_capture_by(x)]];
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
  std::optional<std::string_view> optionalSV;
  vsv.push_back(optionalSV.value());
  vsv.push_back(getOptionalS().value()); // expected-warning {{object whose reference is captured by 'vsv'}}
  vsv.push_back(getOptionalSV().value());
  vsv.push_back(getOptionalMySV().value());

  // (maybe) FIXME: We may choose to diagnose the following case.
  // This happens because 'MyStringViewNotPointer' is not marked as a [[gsl::Pointer]] but is derived from one.
  vsv.push_back(getOptionalMySVNotP().value()); // expected-warning {{object whose reference is captured by 'vsv'}}
}
namespace temporary_views {
void capture1(std::string_view s [[clang::lifetime_capture_by(x)]], std::vector<std::string_view>& x);

// Intended to capture the "string_view" itself
void capture2(const std::string_view& s [[clang::lifetime_capture_by(x)]], std::vector<std::string_view*>& x);
// Intended to capture the pointee of the "string_view"
void capture3(const std::string_view& s [[clang::lifetime_capture_by(x)]], std::vector<std::string_view>& x);

void test1() {
  std::vector<std::string_view> x1;
  capture1(std::string(), x1); // expected-warning {{captured by 'x1'}}
  capture1(std::string_view(), x1);

  std::vector<std::string_view*> x2;
  capture2(std::string_view(), x2); // FIXME: Warn when the temporary view itself is captured.
  capture2(std::string(), x2); // expected-warning {{captured by 'x2'}}
  
  std::vector<std::string_view> x3;
  capture3(std::string_view(), x3);
  capture3(std::string(), x3); // expected-warning {{captured by 'x3'}}
}
}
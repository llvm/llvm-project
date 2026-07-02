// RUN: %clang_cc1 --std=c++20 -fsyntax-only -verify -Wdangling-capture %s
// RUN: %clang_cc1 --std=c++20 -fsyntax-only -Wno-dangling -verify=cfg -Wlifetime-safety %s

#include "Inputs/lifetime-analysis.h"

// ****************************************************************************
// Capture an integer
// ****************************************************************************
namespace capture_int {
struct X {} x;
void captureInt(const int &i [[clang::lifetime_capture_by(x)]], X &x);
void captureRValInt(int &&i [[clang::lifetime_capture_by(x)]], X &x);
void noCaptureInt(int i [[clang::lifetime_capture_by(x)]], X &x);

void temporary_int_capture() {
  captureInt(1,x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                   // cfg-warning {{temporary object does not live long enough}} \
                   // cfg-note {{destroyed here}}
  (void)x;         // cfg-note {{later used here}} 
  captureRValInt(1, x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                        // cfg-warning {{temporary object does not live long enough}} \
                        // cfg-note {{destroyed here}}
( void)x;               // cfg-note {{later used here}} 
}

void local_int_capture() {
  {
    int local;
    captureInt(local, x); // cfg-warning {{local variable 'local' does not live long enough}}
  }                       // cfg-note {{destroyed here}}
  (void)x;                // cfg-note {{later used here}} 
}

void safe_int_captures() {
  noCaptureInt(1, x);
  int local;
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
void noCaptureString(std::string s [[clang::lifetime_capture_by(x)]], X &x);

void temporary_string_capture() {
  captureString(std::string(), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                   // cfg-warning {{temporary object does not live long enough}} \
                                   // cfg-note {{destroyed here}}
  (void)x;                         // cfg-note {{later used here}}
  captureRValString(std::string(), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                       // cfg-warning {{temporary object does not live long enough}} \
                                       // cfg-note {{destroyed here}}
  (void)x;                             // cfg-note {{later used here}}   
}

void local_string_capture() {
  {
    std::string local_string1, local_string2;
    captureString(local_string1, x);                // cfg-warning {{local variable 'local_string1' does not live long enough}}
    captureRValString(std::move(local_string2), x); // cfg-warning {{local variable 'local_string2' does not live long enough}} \
                                    // cfg-note {{result of call to 'move<std::basic_string<char> &>' aliases the storage of local variable 'local_string2'}}
  }                                 // cfg-note 2 {{destroyed here}}                                 
  (void)x;                          // cfg-note 2 {{later used here}}                         
}

void safe_string_captures() {
  noCaptureString(std::string(), x);
  std::string local;
  noCaptureString(local, x);
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

void temporary_string_capture() {
  captureStringView(std::string(), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                       // cfg-warning {{temporary object does not live long enough}} \
                                       // cfg-note {{destroyed here}}
  (void)x;                             // cfg-note {{later used here}}       
  captureRValStringView(std::string(), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                           // cfg-warning {{temporary object does not live long enough}} \
                                           // cfg-note {{destroyed here}}
  (void)x;                                 // cfg-note {{later used here}}                                                      
}

void local_string_capture() {
  {
    std::string local_string;
    captureStringView(getLifetimeBoundView(local_string), x);  // cfg-warning {{local variable 'local_string' does not live long enough}} \
                                                               // cfg-note {{result of call to 'getLifetimeBoundView' aliases the storage of local variable 'local_string'}}
  }                                                            // cfg-note {{destroyed here}}
  (void)x;      // cfg-note {{later used here}}
}

// Lifetimebound captures
void temporary_string_view_lifetimebound_capture() {
  captureStringView(getLifetimeBoundView( // cfg-note {{result of call to 'getLifetimeBoundView' aliases the storage of temporary object}}
  std::string()), x);                     // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                          // cfg-warning {{temporary object does not live long enough}} \
                                          // cfg-note {{destroyed here}}
  (void)x;                                // cfg-note {{later used here}}
  captureStringView(getLifetimeBoundString(std::string()), x);   // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                                                 // cfg-warning {{temporary object does not live long enough}} \
                                                                 // cfg-note {{destroyed here}} \
                                                                 // cfg-note {{result of call to 'getLifetimeBoundString' aliases the storage of temporary object}}
  (void)x;                                                       // cfg-note {{later used here}}
  captureRValStringView(getLifetimeBoundView(std::string()), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                                                 // cfg-warning {{temporary object does not live long enough}} \
                                                                 // cfg-note {{result of call to 'getLifetimeBoundView' aliases the storage of temporary object}} \
                                                                 // cfg-note {{destroyed here}}
  (void)x;                                                       // cfg-note {{later used here}}                                                               
}

void local_string_lifetimebound_capture() {
 {
    std::string local_string;
    captureRValStringView(getLifetimeBoundView(local_string), x); // cfg-warning {{local variable 'local_string' does not live long enough}} \
                                                                  // cfg-note {{result of call to 'getLifetimeBoundView' aliases the storage of local variable 'local_string'}}                                                                                                                        
 }                                                                // cfg-note {{destroyed here}}
 (void)x;                                                         // cfg-note {{later used here}}
}

void temporary_nested_lifetimebound_capture() {
  captureStringView(getLifetimeBoundString(getLifetimeBoundView(std::string())), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                                                                     // cfg-warning {{temporary object does not live long enough}} \
                                                                                     // cfg-note {{destroyed here}} \
                                                                                     // cfg-note {{result of call to 'getLifetimeBoundView' aliases the storage of temporary object}} \
                                                                                     // cfg-note {{result of call to 'getLifetimeBoundString' aliases the storage of temporary object}}
  (void)x;                                                                           // cfg-note {{later used here}}
  captureStringView(getLifetimeBoundString(getLifetimeBoundString(                   // cfg-note 2 {{result of call to 'getLifetimeBoundString' aliases the storage of temporary object}}
    std::string())), x);  // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                          // cfg-warning {{temporary object does not live long enough}} \
                          // cfg-note {{destroyed here}}
  (void)x;                // cfg-note {{later used here}}
}

void safe_captures() {
  std::string_view local_string_view;
  captureStringView(local_string_view, x);
  captureRValStringView(std::move(local_string_view), x);
  captureRValStringView(std::string_view{"abcd"}, x);    
  captureStringView(getNotLifetimeBoundView(std::string()), x);
  captureRValStringView(getNotLifetimeBoundView(std::string()), x);
  noCaptureStringView(local_string_view, x);
  noCaptureStringView(std::string(), x);
  noCaptureStringView(getLifetimeBoundView(std::string()), x);
}
} // namespace capture_string_view

// ****************************************************************************
// Capture pointer (eg: std::string*)
// ****************************************************************************
const std::string* getLifetimeBoundPointer(const std::string &s [[clang::lifetimebound]]);
const std::string* getNotLifetimeBoundPointer(const std::string &s);

namespace capture_pointer {
struct X {} x;   // cfg-note {{this global dangles}}
void capturePointer(const std::string* sp [[clang::lifetime_capture_by(x)]], X &x);

void temporary_pointer_lifetimebound_capture() {
  capturePointer(getLifetimeBoundPointer(std::string()), x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                                             // cfg-warning {{temporary object does not live long enough}} \
                                                             // cfg-note {{destroyed here}} \
                                                             // cfg-note {{result of call to 'getLifetimeBoundPointer' aliases the storage of temporary object}}
  (void)x;                                                   // cfg-note {{later used here}}
}

void temporary_nested_lifetimebound_capture() {
  capturePointer(getLifetimeBoundPointer(*getLifetimeBoundPointer(
    std::string())), x);  // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                          // cfg-warning {{stack memory associated with temporary object escapes to the global variable 'x' which will dangle}}
}

void safe_capture() {
  capturePointer(getNotLifetimeBoundPointer(std::string()), x);
}
} // namespace capture_pointer

// ****************************************************************************
// Arrays and initializer lists.
// ****************************************************************************
namespace init_lists {
struct X {} x;    // cfg-note {{this global dangles}}
void captureVector(const std::vector<int> &a [[clang::lifetime_capture_by(x)]], X &x);
void captureArray(int array [[clang::lifetime_capture_by(x)]] [2], X &x);
void captureInitList(std::initializer_list<int> abc [[clang::lifetime_capture_by(x)]], X &x);

std::initializer_list<int> getLifetimeBoundInitList(std::initializer_list<int> abc [[clang::lifetimebound]]);

void temporary_vector_capture() {
  captureVector({1, 2, 3}, x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                               // cfg-warning {{temporary object does not live long enough}} \
                               // cfg-note {{destroyed here}}
  (void)x;                     // cfg-note {{later used here}}       
  captureVector(std::vector<int>{}, x); // expected-warning {{object whose reference is captured by 'x' will be destroyed at the end of the full-expression}} \
                                        // cfg-warning {{temporary object does not live long enough}} \
                                        // cfg-note {{destroyed here}}
  (void)x;                              // cfg-note {{later used here}}        
}

void local_vector_capture() {
  {
    std::vector<int> local_vector;
    captureVector(local_vector, x);    // cfg-warning {{local variable 'local_vector' does not live long enough}}
  }                                    // cfg-note {{destroyed here}}
  (void)x;                             // cfg-note {{later used here}}
}

void local_array_capture() {
  int local_array[2]; 
  captureArray(local_array, x);      // cfg-warning {{stack memory associated with local variable 'local_array' escapes to the global variable 'x' which will dangle}}
}

// FIXME: Add support for initializer lists in -Wlifetime-safety
void initializer_list_capture() {
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

// FIXME: Add support for capture of method declarations in -Wlifetime-safety
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

// FIXME: Add support for capture by global and unknown in -Wlifetime-safety
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

void temporary_capture_by_this() {
  S s;
  s.captureInt(1); // expected-warning {{object whose reference is captured by 's' will be destroyed at the end of the full-expression}} \
                   // cfg-warning {{temporary object does not live long enough}} \
                   // cfg-note {{destroyed here}}
  (void)s;         // cfg-note {{later used here}}       
  s.captureView(std::string()); // expected-warning {{object whose reference is captured by 's' will be destroyed at the end of the full-expression}} \
                                // cfg-warning {{temporary object does not live long enough}} \
                                // cfg-note {{destroyed here}}
  (void)s;                      // cfg-note {{later used here}}                              
}

void lifetimebound_capture_by_this() {
  S s;
  s.captureView(getLifetimeBoundView(std::string()));    // expected-warning {{object whose reference is captured by 's' will be destroyed at the end of the full-expression}} \
                                                         // cfg-warning {{temporary object does not live long enough}} \
                                                         // cfg-note {{destroyed here}} \
                                                         // cfg-note {{esult of call to 'getLifetimeBoundView' aliases the storage of temporary object}}
  (void)s;                                               // cfg-note {{later used here}}         
  s.captureView(getLifetimeBoundString(std::string()));  // expected-warning {{object whose reference is captured by 's' will be destroyed at the end of the full-expression}} \
                                                         // cfg-warning {{temporary object does not live long enough}} \
                                                         // cfg-note {{destroyed here}} \
                                                         // cfg-note {{result of call to 'getLifetimeBoundString' aliases the storage of temporary object}}
  (void)s;                                               // cfg-note {{later used here}}        
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
  set_of_int.insert(1); // expected-warning {{object whose reference is captured by 'set_of_int' will be destroyed at the end of the full-expression}} \
                        // cfg-warning {{temporary object does not live long enough}} \
                        // cfg-note {{destroyed here}}
  (void)set_of_int;     // cfg-note {{later used here}}                   
  MySet<std::string_view> set_of_sv;
  set_of_sv.insert(std::string());       // expected-warning {{object whose reference is captured by 'set_of_sv' will be destroyed at the end of the full-expression}} \
                                         // cfg-warning {{temporary object does not live long enough}} \
                                         // cfg-note {{destroyed here}}
  (void)set_of_sv;                       // cfg-note {{later used here}}                
  set_of_sv.insert(std::string_view());
  (void)set_of_sv;                                             
                                    
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
  vector_of_view.push_back(std::string()); // expected-warning {{object whose reference is captured by 'vector_of_view' will be destroyed at the end of the full-expression}} \
                                           // cfg-warning {{temporary object does not live long enough}} \
                                           // cfg-note {{destroyed here}}
  (void)vector_of_view;                    // cfg-note {{later used here}}
  vector_of_view.push_back(getLifetimeBoundView(std::string())); // expected-warning {{object whose reference is captured by 'vector_of_view' will be destroyed at the end of the full-expression}} \
                                                                 // cfg-warning {{temporary object does not live long enough}} \
                                                                 // cfg-note {{destroyed here}} \
                                                                 // cfg-note {{result of call to 'getLifetimeBoundView' aliases the storage of temporary object}}
  (void)vector_of_view;                                          // cfg-note {{later used here}}

  MyVector<const std::string*> vector_of_pointer;
  vector_of_pointer.push_back(getLifetimeBoundPointer(std::string())); // expected-warning {{object whose reference is captured by 'vector_of_pointer' will be destroyed at the end of the full-expression}} \
                                                                       // cfg-warning {{temporary object does not live long enough}} \
                                                                       // cfg-note {{destroyed here}}
  (void)vector_of_pointer;                                             // cfg-note {{later used here}}
  vector_of_pointer.push_back(getLifetimeBoundPointer(*getLifetimeBoundPointer(std::string()))); // expected-warning {{object whose reference is captured by 'vector_of_pointer' will be destroyed at the end of the full-expression}} \
                                                                                                 // cfg-warning {{temporary object does not live long enough}} \
                                                                                                 // cfg-note {{destroyed here}}
  (void)vector_of_pointer;                                                                       // cfg-note {{later used here}}
  vector_of_pointer.push_back(getLifetimeBoundPointer(local));              // cfg-warning {{temporary object does not live long enough}} \
                                                                            // cfg-note {{destroyed here}}
  (void)vector_of_pointer;                                                  // cfg-note {{later used here}}
  vector_of_pointer.push_back(getNotLifetimeBoundPointer(std::string()));   // cfg-warning {{temporary object does not live long enough}} \
                                                                            // cfg-note {{destroyed here}}
  (void)vector_of_pointer;                                                  // cfg-note {{later used here}}
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
  (void)vector_of_my_view;
  vector_of_my_view.push_back(MyStringView{});
  (void)vector_of_my_view;
  vector_of_my_view.push_back(std::string_view{});
  (void)vector_of_my_view;
  vector_of_my_view.push_back(std::string{});       // expected-warning {{object whose reference is captured by 'vector_of_my_view' will be destroyed at the end of the full-expression}} \
                                                    // cfg-warning {{temporary object does not live long enough}} \
                                                    // cfg-note {{destroyed here}}
  (void)vector_of_my_view;                          // cfg-note {{later used here}}                                            
  vector_of_my_view.push_back(getLifetimeBoundView(std::string{})); // expected-warning {{object whose reference is captured by 'vector_of_my_view' will be destroyed at the end of the full-expression}} \
                                                    // cfg-warning {{temporary object does not live long enough}} \
                                                    // cfg-note {{destroyed here}} \
                                                    // cfg-note {{result of call to 'getLifetimeBoundView' aliases the storage of temporary object}}
  (void)vector_of_my_view;                          // cfg-note {{later used here}}                                                   
  vector_of_my_view.push_back(getLifetimeBoundString(getLifetimeBoundView(std::string{}))); // expected-warning {{object whose reference is captured by 'vector_of_my_view' will be destroyed at the end of the full-expression}} \
                                                    // cfg-warning {{temporary object does not live long enough}} \
                                                    // cfg-note {{destroyed here}} \
                                                    // cfg-note {{esult of call to 'getLifetimeBoundView' aliases the storage of temporary object}} \
                                                    // cfg-note {{result of call to 'getLifetimeBoundString' aliases the storage of temporary object}}
  (void)vector_of_my_view;                          // cfg-note {{later used here}}                                                    
  vector_of_my_view.push_back(getNotLifetimeBoundView(getLifetimeBoundString(getLifetimeBoundView(std::string{}))));
  (void)vector_of_my_view;
  
  // Use with container of other view types.
  MyVector<std::string_view> vector_of_view;
  vector_of_view.push_back(getMySV());
  (void)vector_of_view;
  vector_of_view.push_back(getMySVNotP());
  (void)vector_of_view;
}

// ****************************************************************************
// Container: Use with std::optional<view> (owner<pointer> types)
// ****************************************************************************
void use_with_optional_view() {
  MyVector<std::string_view> vector_of_view;

  std::optional<std::string_view> optional_of_view;
  vector_of_view.push_back(optional_of_view.value());
  vector_of_view.push_back(getOptionalS().value());      // expected-warning {{object whose reference is captured by 'vector_of_view' will be destroyed at the end of the full-expression}} \
                                                         // cfg-warning {{temporary object does not live long enough}} \
                                                         // cfg-note {{destroyed here}} \
                                                         // cfg-note {{result of call to 'value' aliases the storage of temporary object}}                                                         
  (void)vector_of_view;                                  // cfg-note {{later used here}}  
  vector_of_view.push_back(getOptionalSV().value());
  (void)vector_of_view;
  vector_of_view.push_back(getOptionalMySV().value());
  (void)vector_of_view;
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
  capture1(std::string(), x1); // expected-warning {{object whose reference is captured by 'x1' will be destroyed at the end of the full-expression}} \
                               // cfg-warning {{temporary object does not live long enough}} \
                               // cfg-note {{destroyed here}}
  (void)x1;                    // cfg-note {{later used here}}
  capture1(std::string_view(), x1);

  std::vector<std::string_view*> x2;
  // Clang considers 'const std::string_view&' to refer to the owner
  // 'std::string' and not 'std::string_view'. Therefore no diagnostic here.
  capture2(std::string_view(), x2);
  (void)x2;
  capture2(std::string(), x2);       // expected-warning {{object whose reference is captured by 'x2' will be destroyed at the end of the full-expression}} \
                                     // cfg-warning {{temporary object does not live long enough}} \
                                     // cfg-note {{destroyed here}}
  (void)x2;                          // cfg-note {{later used here}}
  
  std::vector<std::string_view> x3;
  capture3(std::string_view(), x3);
  (void)x3;      
  capture3(std::string(), x3);       // expected-warning {{object whose reference is captured by 'x3' will be destroyed at the end of the full-expression}} \
                                     // cfg-warning {{temporary object does not live long enough}}  \
                                     // cfg-note {{destroyed here}}
  (void)x3;                          // cfg-note {{later used here}}       
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
  std::vector<std::string_view> views;
  views.push_back(std::string()); // expected-warning {{object whose reference is captured by 'views' will be destroyed at the end of the full-expression}} \
                                  // cfg-warning {{temporary object does not live long enough}} \
                                  // cfg-note {{destroyed here}}
  (void)views;                    // cfg-note {{later used here}}             
  views.insert(views.begin(), 
            std::string());       // expected-warning {{object whose reference is captured by 'views' will be destroyed at the end of the full-expression}} \
                                  // cfg-warning {{temporary object does not live long enough}} \
                                  // cfg-note {{destroyed here}}
  (void)views;                    // cfg-note {{later used here}}                             
  views.push_back(getLifetimeBoundView(std::string()));    // expected-warning {{object whose reference is captured by 'views' will be destroyed at the end of the full-expression}} \
                                                           // cfg-warning {{temporary object does not live long enough}} \
                                                           // cfg-note {{destroyed here}} \
                                                           // cfg-note {{result of call to 'getLifetimeBoundView' aliases the storage of temporary object}}
  (void)views;                                             // cfg-note {{later used here}}                                                             
  views.push_back(getNotLifetimeBoundView(std::string()));
  (void)views;
  {
      std::string local1, local2;
      views.push_back(local1);    // cfg-warning {{local variable 'local1' does not live long enough}}
      (void)views;       
      views.insert(views.end(), local2);  // cfg-warning {{local variable 'local2' does not live long enough}}
  }                                       // cfg-note 2 {{destroyed here}}                                       
  (void)views;                            // cfg-note 2 {{later used here}}

  std::vector<std::string> strings;
  strings.push_back(std::string());
  (void)views;
  strings.insert(strings.begin(), std::string());

  std::vector<const std::string*> pointers;
  pointers.push_back(getLifetimeBoundPointer(std::string()));
  (void)views;
  std::string local;
  pointers.push_back(&local);
  (void)views;
}

namespace with_span {
// Templated view types.
template<typename T>
struct [[gsl::Pointer]] Span {
  Span(const std::vector<T> &V);
};

void use() {
  std::vector<Span<int>> spans;
  spans.push_back(std::vector<int>{1, 2, 3}); // expected-warning {{object whose reference is captured by 'spans' will be destroyed at the end of the full-expression}} \
                                              // cfg-warning {{temporary object does not live long enough}} \
                                              // cfg-note {{destroyed here}}
  (void)spans;                                // cfg-note {{later used here}}                                    
  {
    std::vector<int> local;
    spans.push_back(local);    // cfg-warning {{local variable 'local' does not live long enough}}         
  }                            // cfg-note {{destroyed here}}                        
  (void)spans;                 // cfg-note {{later used here}} 
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

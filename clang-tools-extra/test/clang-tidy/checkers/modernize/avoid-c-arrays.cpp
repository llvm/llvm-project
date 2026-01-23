// RUN: %check_clang_tidy -std=c++17 %s modernize-avoid-c-arrays %t

int a[] = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead

int b[1];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead

void foo() {
  int c[b[0]];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C VLA arrays, use 'std::vector' instead

  using d = decltype(c);
  d e;
  // Semi-FIXME: we do not diagnose these last two lines separately,
  // because we point at typeLoc.getBeginLoc(), which is the decl before that
  // (int c[b[0]];), which is already diagnosed.
}

template <typename T, int Size>
class array {
  T d[Size];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead

  int e[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead
};

array<int[4], 2> d;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead

using k = int[4];
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not declare C-style arrays, use 'std::array' instead

k ak;
// no diagnostic expected here since no concrete C-style array type is written here

array<k, 2> dk;
// no diagnostic expected here since no concrete C-style array type is written here

array<decltype(ak), 3> ek;
// no diagnostic expected here since no concrete C-style array type is written here

template <typename T>
class unique_ptr {
  T *d;

  int e[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead
};

unique_ptr<int[]> d2;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use 'std::array' instead

using k2 = int[];
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use 'std::array' instead

unique_ptr<k2> dk2;

// Some header
extern "C" {

int f[] = {1, 2};

int j[1];

inline void bar() {
  {
    int j[j[0]];
  }
}

extern "C++" {
int f3[] = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead

int j3[1];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead

struct Foo {
  int f3[3] = {1, 2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead

  int j3[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead
};
}

struct Bar {

  int f[3] = {1, 2};

  int j[1];
};
}

const char name[] = "Some string";
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

void takeCharArray(const char name[]);
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: do not declare C-style arrays, use 'std::array' or 'std::vector' instead [modernize-avoid-c-arrays]

namespace std {
  template<class T, class U>
  struct is_same { constexpr static bool value{false}; };

  template<class T>
  struct is_same<T, T> { constexpr static bool value{true}; };

  template<class T, class U>
  constexpr bool is_same_v = is_same<T, U>::value;

  template<class T> struct remove_const { typedef T type; };
  template<class T> struct remove_const<const T> { typedef T type; };

  template<class T>
  using remove_const_t = typename remove_const<T>::type;

  template<bool B, class T = void> struct enable_if {};
  template<class T> struct enable_if<true, T> { typedef T type; };

  template< bool B, class T = void >
  using enable_if_t = typename enable_if<B, T>::type;
}

// within below template decl, no array type findings are expected within the template parameter declarations since not a single C-style array type got written explicitly
template <typename T,
          bool = std::is_same_v<T, int>,
          bool = std::is_same<T, int>::value,
          bool = std::is_same_v<std::remove_const_t<T>, int>,
          bool = std::is_same<std::remove_const_t<T>, int>::value,
          bool = std::is_same_v<typename std::remove_const<T>::type, int>,
          bool = std::is_same<typename std::remove_const<T>::type, int>::value,
          std::enable_if_t<not(std::is_same_v<std::remove_const_t<T>, int>) && not(std::is_same_v<typename std::remove_const<T>::type, char>), bool> = true,
          typename std::enable_if<not(std::is_same_v<std::remove_const_t<T>, int>) && not(std::is_same_v<typename std::remove_const<T>::type, char>), bool>::type = true,
          typename = std::enable_if_t<not(std::is_same_v<std::remove_const_t<T>, int>) && not(std::is_same_v<typename std::remove_const<T>::type, char>)>,
          typename = typename std::remove_const<T>::type,
          typename = std::remove_const_t<T>>
class MyClassTemplate {
 public:
  // here, however, plenty of array type findings are expected for below template parameter declarations since C-style array types are written explicitly
  template <typename U = T,
            bool = std::is_same_v<U, int[]>,
            // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
            bool = std::is_same<U, int[10]>::value,
            // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
            std::enable_if_t<not(std::is_same_v<std::remove_const_t<U>, int[]>) && not(std::is_same_v<typename std::remove_const<U>::type, char[10]>), bool> = true,
            // CHECK-MESSAGES: :[[@LINE-1]]:73: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
            // CHECK-MESSAGES: :[[@LINE-2]]:140: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
            typename = typename std::remove_const<int[10]>::type,
            // CHECK-MESSAGES: :[[@LINE-1]]:51: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
            typename = std::remove_const_t<int[]>>
            // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
    class MyInnerClassTemplate {
     public:
      MyInnerClassTemplate(const U&) {}
     private:
      U field[3];
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
    };

    MyClassTemplate(const T&) {}

 private:
    T field[7];
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
};

// an explicit instantiation
template
class MyClassTemplate<int[2]>;
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

using MyArrayType = int[3];
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

// another explicit instantiation
template
class MyClassTemplate<MyArrayType>;
// no diagnostic is expected here since no C-style array type got written here

// within below template decl, no array type findings are expected within the template parameter declarations since not a single C-style array type got written explicitly
template <typename T,
          bool = std::is_same_v<T, int>,
          bool = std::is_same<T, int>::value,
          bool = std::is_same_v<std::remove_const_t<T>, int>,
          bool = std::is_same<std::remove_const_t<T>, int>::value,
          bool = std::is_same_v<typename std::remove_const<T>::type, int>,
          bool = std::is_same<typename std::remove_const<T>::type, int>::value,
          std::enable_if_t<not(std::is_same_v<std::remove_const_t<T>, int>) && not(std::is_same_v<typename std::remove_const<T>::type, char>), bool> = true,
          typename std::enable_if<not(std::is_same_v<std::remove_const_t<T>, int>) && not(std::is_same_v<typename std::remove_const<T>::type, char>), bool>::type = true,
          typename = std::enable_if_t<not(std::is_same_v<std::remove_const_t<T>, int>) && not(std::is_same_v<typename std::remove_const<T>::type, char>)>,
          typename = typename std::remove_const<T>::type,
          typename = std::remove_const_t<T>>
void func(const T& param) {
  int array1[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

  T array2[2];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

  T value;
}

// here, however, plenty of array type findings are expected for below template parameter declarations since C-style array types are written explicitly
template <typename T = int[],
          // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
          bool = std::is_same_v<T, int[]>,
          // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
          bool = std::is_same<T, int[10]>::value,
          // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
          std::enable_if_t<not(std::is_same_v<std::remove_const_t<T>, int[]>) && not(std::is_same_v<typename std::remove_const<T>::type, char[10]>), bool> = true,
          // CHECK-MESSAGES: :[[@LINE-1]]:71: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
          // CHECK-MESSAGES: :[[@LINE-2]]:138: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
          typename = typename std::remove_const<int[10]>::type,
          // CHECK-MESSAGES: :[[@LINE-1]]:49: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
          typename = std::remove_const_t<int[]>>
          // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
void fun(const T& param) {
  int array3[3];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

  T array4[4];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

  T value;
}

template<typename T>
T some_constant{};

// explicit instantiations
template
int some_constant<int[5]>[5];
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

template
int some_constant<decltype(ak)>[4];
// no diagnostic is expected here since explicit instantiations aren't represented as `TypeLoc` in the AST and we hence cannot match them as such

MyArrayType mk;
// no diagnostic is expected here since no C-style array type got written here

// explicit specializations
template<>
int some_constant<int[7]>[7]{};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

template<>
int some_constant<decltype(mk)>[3]{};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

void testArrayInTemplateType() {
  int t[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

  func(t);
  fun(t);

  func<decltype(t)>({});
  fun<decltype(t)>({});

  func<int[1]>({});
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
  fun<int[1]>({});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

  MyClassTemplate var{t};
  MyClassTemplate<decltype(t)> var1{{}};
  MyClassTemplate<int[2]> var2{{}};
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

  decltype(var1)::MyInnerClassTemplate var3{t};
  decltype(var1)::MyInnerClassTemplate<decltype(t)> var4{{}};
  decltype(var1)::MyInnerClassTemplate<char[5]> var5{{}};
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

  MyClassTemplate<decltype(t)>::MyInnerClassTemplate var6{t};
  MyClassTemplate<decltype(t)>::MyInnerClassTemplate<decltype(t)> var7{{}};
  MyClassTemplate<decltype(t)>::MyInnerClassTemplate<char[8]> var8{{}};
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
  MyClassTemplate<int[9]>::MyInnerClassTemplate<char[9]> var9{{}};
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
  // CHECK-MESSAGES: :[[@LINE-2]]:49: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]
}

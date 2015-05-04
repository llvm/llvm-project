// RUN: not %clang_cc1 -fsyntax-only %s -std=c++11 2>&1 | FileCheck %s -check-prefix=CHECK-ELIDE-NOTREE
// RUN: not %clang_cc1 -fsyntax-only %s -fno-elide-type -std=c++11 2>&1 | FileCheck %s -check-prefix=CHECK-NOELIDE-NOTREE
// RUN: not %clang_cc1 -fsyntax-only %s -fdiagnostics-show-template-tree -std=c++11 2>&1 | FileCheck %s -check-prefix=CHECK-ELIDE-TREE
// RUN: not %clang_cc1 -fsyntax-only %s -fno-elide-type -fdiagnostics-show-template-tree -std=c++11 2>&1 | FileCheck %s -check-prefix=CHECK-NOELIDE-TREE

// PR9548 - "no known conversion from 'vector<string>' to 'vector<string>'"
// vector<string> refers to two different types here.  Make sure the message
// gives a way to tell them apart.
class versa_string;
typedef versa_string string;

namespace std {template <typename T> class vector;}
using std::vector;

void f(vector<string> v);

namespace std {
  class basic_string;
  typedef basic_string string;
  template <typename T> class vector {};
  void g() {
    vector<string> v;
    f(v);
  }
} // end namespace std
// CHECK-ELIDE-NOTREE: no matching function for call to 'f'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<std::basic_string>' to 'vector<versa_string>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<std::basic_string>' to 'vector<versa_string>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   vector<
// CHECK-ELIDE-TREE:     [std::basic_string != versa_string]>
// CHECK-NOELIDE-TREE: no matching function for call to 'f'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   vector<
// CHECK-NOELIDE-TREE:     [std::basic_string != versa_string]>

template <int... A>
class I1{};
void set1(I1<1,2,3,4,2,3,4,3>) {};
void test1() {
  set1(I1<1,2,3,4,2,2,4,3,7>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set1'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'I1<[5 * ...], 2, [2 * ...], 7>' to 'I1<[5 * ...], 3, [2 * ...], (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set1'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'I1<1, 2, 3, 4, 2, 2, 4, 3, 7>' to 'I1<1, 2, 3, 4, 2, 3, 4, 3, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set1'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   I1<
// CHECK-ELIDE-TREE:     [5 * ...],
// CHECK-ELIDE-TREE:     [2 != 3],
// CHECK-ELIDE-TREE:     [2 * ...],
// CHECK-ELIDE-TREE:     [7 != (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set1'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   I1<
// CHECK-NOELIDE-TREE:     1,
// CHECK-NOELIDE-TREE:     2,
// CHECK-NOELIDE-TREE:     3,
// CHECK-NOELIDE-TREE:     4,
// CHECK-NOELIDE-TREE:     2,
// CHECK-NOELIDE-TREE:     [2 != 3],
// CHECK-NOELIDE-TREE:     4,
// CHECK-NOELIDE-TREE:     3,
// CHECK-NOELIDE-TREE:     [7 != (no argument)]>

template <class A, class B, class C = void>
class I2{};
void set2(I2<int, int>) {};
void test2() {
  set2(I2<double, int, int>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set2'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'I2<double, [...], int>' to 'I2<int, [...], (default) void>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set2'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'I2<double, int, int>' to 'I2<int, int, (default) void>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set2'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   I2<
// CHECK-ELIDE-TREE:     [double != int],
// CHECK-ELIDE-TREE:     [...], 
// CHECK-ELIDE-TREE:     [int != (default) void]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set2'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   I2<
// CHECK-NOELIDE-TREE:     [double != int],
// CHECK-NOELIDE-TREE:     int,
// CHECK-NOELIDE-TREE:     [int != (default) void]>

int V1, V2, V3;
template <int* A, int *B>
class I3{};
void set3(I3<&V1, &V2>) {};
void test3() {
  set3(I3<&V3, &V2>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set3'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'I3<&V3, [...]>' to 'I3<&V1, [...]>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set3'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'I3<&V3, &V2>' to 'I3<&V1, &V2>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set3'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   I3<
// CHECK-ELIDE-TREE:     [&V3 != &V1]
// CHECK-ELIDE-TREE:     [...]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set3'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   I3<
// CHECK-NOELIDE-TREE:     [&V3 != &V1]
// CHECK-NOELIDE-TREE:     &V2>

template <class A, class B>
class Alpha{};
template <class A, class B>
class Beta{};
template <class A, class B>
class Gamma{};
template <class A, class B>
class Delta{};

void set4(Alpha<int, int>);
void test4() {
  set4(Beta<void, void>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set4'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'Beta<void, void>' to 'Alpha<int, int>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set4'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'Beta<void, void>' to 'Alpha<int, int>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set4'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from 'Beta<void, void>' to 'Alpha<int, int>' for 1st argument
// CHECK-NOELIDE-TREE: no matching function for call to 'set4'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from 'Beta<void, void>' to 'Alpha<int, int>' for 1st argument

void set5(Alpha<Beta<Gamma<Delta<int, int>, int>, int>, int>);
void test5() {
  set5(Alpha<Beta<Gamma<void, void>, double>, double>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set5'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'Alpha<Beta<Gamma<void, void>, double>, double>' to 'Alpha<Beta<Gamma<Delta<int, int>, int>, int>, int>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set5'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'Alpha<Beta<Gamma<void, void>, double>, double>' to 'Alpha<Beta<Gamma<Delta<int, int>, int>, int>, int>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set5'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Alpha<
// CHECK-ELIDE-TREE:     Beta<
// CHECK-ELIDE-TREE:       Gamma<
// CHECK-ELIDE-TREE:         [void != Delta<int, int>],
// CHECK-ELIDE-TREE:         [void != int]>
// CHECK-ELIDE-TREE:       [double != int]>
// CHECK-ELIDE-TREE:     [double != int]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set5'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Alpha<
// CHECK-NOELIDE-TREE:     Beta<
// CHECK-NOELIDE-TREE:       Gamma<
// CHECK-NOELIDE-TREE:         [void != Delta<int, int>],
// CHECK-NOELIDE-TREE:         [void != int]>
// CHECK-NOELIDE-TREE:       [double != int]>
// CHECK-NOELIDE-TREE:     [double != int]>

void test6() {
  set5(Alpha<Beta<Delta<int, int>, int>, int>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set5'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'Alpha<Beta<Delta<int, int>, [...]>, [...]>' to 'Alpha<Beta<Gamma<Delta<int, int>, int>, [...]>, [...]>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set5'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'Alpha<Beta<Delta<int, int>, int>, int>' to 'Alpha<Beta<Gamma<Delta<int, int>, int>, int>, int>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set5'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Alpha<
// CHECK-ELIDE-TREE:     Beta<
// CHECK-ELIDE-TREE:       [Delta<int, int> != Gamma<Delta<int, int>, int>],
// CHECK-ELIDE-TREE:       [...]>
// CHECK-ELIDE-TREE:     [...]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set5'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Alpha<
// CHECK-NOELIDE-TREE:     Beta<
// CHECK-NOELIDE-TREE:       [Delta<int, int> != Gamma<Delta<int, int>, int>],
// CHECK-NOELIDE-TREE:       int>
// CHECK-NOELIDE-TREE:     int>

int a7, b7;
int c7[] = {1,2,3};
template<int *A>
class class7 {};
void set7(class7<&a7> A) {}
void test7() {
  set7(class7<&a7>());
  set7(class7<&b7>());
  set7(class7<c7>());
  set7(class7<nullptr>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<&b7>' to 'class7<&a7>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<c7>' to 'class7<&a7>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<nullptr>' to 'class7<&a7>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<&b7>' to 'class7<&a7>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<c7>' to 'class7<&a7>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set7'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class7<nullptr>' to 'class7<&a7>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set7'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class7<
// CHECK-ELIDE-TREE:     [&b7 != &a7]>
// CHECK-ELIDE-TREE: no matching function for call to 'set7'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class7<
// CHECK-ELIDE-TREE:     [c7 != &a7]>
// CHECK-ELIDE-TREE: no matching function for call to 'set7'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class7<
// CHECK-ELIDE-TREE:     [nullptr != &a7]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set7'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class7<
// CHECK-NOELIDE-TREE:     [&b7 != &a7]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set7'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class7<
// CHECK-NOELIDE-TREE:     [c7 != &a7]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set7'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class7<
// CHECK-NOELIDE-TREE:     [nullptr != &a7]>

template<typename ...T> struct S8 {};
template<typename T> using U8 = S8<int, char, T>;
int f8(S8<int, char, double>);
int k8 = f8(U8<char>());
// CHECK-ELIDE-NOTREE: no matching function for call to 'f8'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'S8<[2 * ...], char>' to 'S8<[2 * ...], double>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f8'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'S8<int, char, char>' to 'S8<int, char, double>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f8'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   S8<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     [char != double]>
// CHECK-NOELIDE-TREE: no matching function for call to 'f8'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   S8<
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     char, 
// CHECK-NOELIDE-TREE:     [char != double]>

template<typename ...T> struct S9 {};
template<typename T> using U9 = S9<int, char, T>;
template<typename T> using V9 = U9<U9<T>>;
int f9(S9<int, char, U9<const double>>);
int k9 = f9(V9<double>());

// CHECK-ELIDE-NOTREE: no matching function for call to 'f9'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'S9<[2 * ...], S9<[2 * ...], double>>' to 'S9<[2 * ...], S9<[2 * ...], const double>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f9'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'S9<int, char, S9<int, char, double>>' to 'S9<int, char, S9<int, char, const double>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f9'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   S9<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     S9<
// CHECK-ELIDE-TREE:       [2 * ...], 
// CHECK-ELIDE-TREE:       [double != const double]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'f9'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   S9<
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     char, 
// CHECK-NOELIDE-TREE:     S9<
// CHECK-NOELIDE-TREE:       int, 
// CHECK-NOELIDE-TREE:       char, 
// CHECK-NOELIDE-TREE:       [double != const double]>>

template<typename ...A> class class_types {};
void set10(class_types<int, int>) {}
void test10() {
  set10(class_types<int>());
  set10(class_types<int, int, int>());
}

// CHECK-ELIDE-NOTREE: no matching function for call to 'set10'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_types<[...], (no argument)>' to 'class_types<[...], int>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set10'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_types<[2 * ...], int>' to 'class_types<[2 * ...], (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set10'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_types<int, (no argument)>' to 'class_types<int, int>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set10'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_types<int, int, int>' to 'class_types<int, int, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set10'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_types<
// CHECK-ELIDE-TREE:     [...], 
// CHECK-ELIDE-TREE:     [(no argument) != int]>
// CHECK-ELIDE-TREE: no matching function for call to 'set10'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_types<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     [int != (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set10'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_types<
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     [(no argument) != int]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set10'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_types<
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     int, 
// CHECK-NOELIDE-TREE:     [int != (no argument)]>

template<int ...A> class class_ints {};
void set11(class_ints<2, 3>) {}
void test11() {
  set11(class_ints<1>());
  set11(class_ints<0, 3, 6>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set11'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ints<1, (no argument)>' to 'class_ints<2, 3>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set11'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ints<0, [...], 6>' to 'class_ints<2, [...], (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set11'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ints<1, (no argument)>' to 'class_ints<2, 3>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set11'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ints<0, 3, 6>' to 'class_ints<2, 3, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set11'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_ints<
// CHECK-ELIDE-TREE:     [1 != 2], 
// CHECK-ELIDE-TREE:     [(no argument) != 3]>
// CHECK-ELIDE-TREE: no matching function for call to 'set11'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_ints<
// CHECK-ELIDE-TREE:     [0 != 2], 
// CHECK-ELIDE-TREE:     [...], 
// CHECK-ELIDE-TREE:     [6 != (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set11'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_ints<
// CHECK-NOELIDE-TREE:     [1 != 2], 
// CHECK-NOELIDE-TREE:     [(no argument) != 3]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set11'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_ints<
// CHECK-NOELIDE-TREE:     [0 != 2], 
// CHECK-NOELIDE-TREE:     3, 
// CHECK-NOELIDE-TREE:     [6 != (no argument)]>

template<template<class> class ...A> class class_template_templates {};
template<class> class tt1 {};
template<class> class tt2 {};
void set12(class_template_templates<tt1, tt1>) {}
void test12() {
  set12(class_template_templates<tt2>());
  set12(class_template_templates<tt1, tt1, tt1>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set12'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_template_templates<template tt2, template (no argument)>' to 'class_template_templates<template tt1, template tt1>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set12'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_template_templates<[2 * ...], template tt1>' to 'class_template_templates<[2 * ...], template (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set12'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_template_templates<template tt2, template (no argument)>' to 'class_template_templates<template tt1, template tt1>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set12'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_template_templates<template tt1, template tt1, template tt1>' to 'class_template_templates<template tt1, template tt1, template (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set12'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_template_templates<
// CHECK-ELIDE-TREE:     [template tt2 != template tt1], 
// CHECK-ELIDE-TREE:     [template (no argument) != template tt1]>
// CHECK-ELIDE-TREE: no matching function for call to 'set12'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_template_templates<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     [template tt1 != template (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set12'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_template_templates<
// CHECK-NOELIDE-TREE:     [template tt2 != template tt1], 
// CHECK-NOELIDE-TREE:     [template (no argument) != template tt1]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set12'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_template_templates<
// CHECK-NOELIDE-TREE:     template tt1, 
// CHECK-NOELIDE-TREE:     template tt1, 
// CHECK-NOELIDE-TREE:     [template tt1 != template (no argument)]>

double a13, b13, c13, d13;
template<double* ...A> class class_ptrs {};
void set13(class_ptrs<&a13, &b13>) {}
void test13() {
  set13(class_ptrs<&c13>());
  set13(class_ptrss<&a13, &b13, &d13>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set13'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ptrs<&c13, (no argument)>' to 'class_ptrs<&a13, &b13>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'set13'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ptrs<[2 * ...], &d13>' to 'class_ptrs<[2 * ...], (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set13'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ptrs<&c13, (no argument)>' to 'class_ptrs<&a13, &b13>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set13'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'class_ptrs<&a13, &b13, &d13>' to 'class_ptrs<&a13, &b13, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set13'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_ptrs<
// CHECK-ELIDE-TREE:     [&c13 != &a13], 
// CHECK-ELIDE-TREE:     [(no argument) != &b13]>
// CHECK-ELIDE-TREE: no matching function for call to 'set13'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   class_ptrs<
// CHECK-ELIDE-TREE:     [2 * ...], 
// CHECK-ELIDE-TREE:     [&d13 != (no argument)]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set13'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_ptrs<
// CHECK-NOELIDE-TREE:     [&c13 != &a13], 
// CHECK-NOELIDE-TREE:     [(no argument) != &b13]>
// CHECK-NOELIDE-TREE: no matching function for call to 'set13'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   class_ptrs<
// CHECK-NOELIDE-TREE:     &a13, 
// CHECK-NOELIDE-TREE:     &b13, 
// CHECK-NOELIDE-TREE:     [&d13 != (no argument)]>

template<typename T> struct s14 {};
template<typename T> using a14 = s14<T>;
typedef a14<int> b14;
template<typename T> using c14 = b14;
int f14(c14<int>);
int k14 = f14(a14<char>());
// CHECK-ELIDE-NOTREE: no matching function for call to 'f14'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'a14<char>' to 'a14<int>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f14'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'a14<char>' to 'a14<int>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f14'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   a14<
// CHECK-ELIDE-TREE:     [char != int]>
// CHECK-NOELIDE-TREE: no matching function for call to 'f14'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   a14<
// CHECK-NOELIDE-TREE:     [char != int]>

void set15(vector<vector<int>>) {}
void test15() {
  set15(vector<vector<int>>());
}
// CHECK-ELIDE-NOTREE-NOT: set15
// CHECK-NOELIDE-NOTREE-NOT: set15
// CHECK-ELIDE-TREE-NOT: set15
// CHECK-NOELIDE-TREE-NOT: set15
// no error here

void set16(vector<const vector<int>>) {}
void test16() {
  set16(vector<const vector<const int>>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set16'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<const vector<const int>>' to 'vector<const vector<int>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set16'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<const vector<const int>>' to 'vector<const vector<int>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set16'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   vector<
// CHECK-ELIDE-TREE:     const vector<
// CHECK-ELIDE-TREE:       [const != (no qualifiers)] int>>
// CHECK-NOELIDE-TREE: no matching function for call to 'set16'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   vector<
// CHECK-NOELIDE-TREE:     const vector<
// CHECK-NOELIDE-TREE:       [const != (no qualifiers)] int>>

void set17(vector<vector<int>>) {}
void test17() {
  set17(vector<const vector<int>>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set17'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<const vector<[...]>>' to 'vector<vector<[...]>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set17'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<const vector<int>>' to 'vector<vector<int>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set17'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   vector<
// CHECK-ELIDE-TREE:     [const != (no qualifiers)] vector<
// CHECK-ELIDE-TREE:       [...]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'set17'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   vector<
// CHECK-NOELIDE-TREE:     [const != (no qualifiers)] vector<
// CHECK-NOELIDE-TREE:       int>>

void set18(vector<const vector<int>>) {}
void test18() {
  set18(vector<vector<int>>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set18'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<vector<[...]>>' to 'vector<const vector<[...]>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set18'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<vector<int>>' to 'vector<const vector<int>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set18'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   vector<
// CHECK-ELIDE-TREE:     [(no qualifiers) != const] vector<
// CHECK-ELIDE-TREE:       [...]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'set18'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   vector<
// CHECK-NOELIDE-TREE:     [(no qualifiers) != const] vector<
// CHECK-NOELIDE-TREE:       int>>

void set19(vector<volatile vector<int>>) {}
void test19() {
  set19(vector<const vector<int>>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set19'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<const vector<[...]>>' to 'vector<volatile vector<[...]>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set19'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<const vector<int>>' to 'vector<volatile vector<int>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set19'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   vector<
// CHECK-ELIDE-TREE:     [const != volatile] vector<
// CHECK-ELIDE-TREE:       [...]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'set19'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   vector<
// CHECK-NOELIDE-TREE:     [const != volatile] vector<
// CHECK-NOELIDE-TREE:       int>>

void set20(vector<const volatile vector<int>>) {}
void test20() {
  set20(vector<const vector<int>>());
}
// CHECK-ELIDE-NOTREE: no matching function for call to 'set20'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<const vector<[...]>>' to 'vector<const volatile vector<[...]>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'set20'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<const vector<int>>' to 'vector<const volatile vector<int>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'set20'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   vector<
// CHECK-ELIDE-TREE:     [const != const volatile] vector<
// CHECK-ELIDE-TREE:       [...]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'set20'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   vector<
// CHECK-NOELIDE-TREE:     [const != const volatile] vector<
// CHECK-NOELIDE-TREE:       int>>


// Checks that volatile does not show up in diagnostics.
template<typename T> struct S21 {};
template<typename T> using U21 = volatile S21<T>;
int f21(vector<const U21<int>>);
int k21 = f21(vector<U21<int>>());
// CHECK-ELIDE-NOTREE: no matching function for call to 'f21'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<U21<[...]>>' to 'vector<const U21<[...]>>' for 1st argument 
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f21'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<U21<int>>' to 'vector<const U21<int>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f21'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:    vector<
// CHECK-ELIDE-TREE:      [(no qualifiers) != const] U21<
// CHECK-ELIDE-TREE:        [...]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'f21'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:    vector<
// CHECK-NOELIDE-TREE:      [(no qualifiers) != const] U21<
// CHECK-NOELIDE-TREE:        int>>

// Checks that volatile does not show up in diagnostics.
template<typename T> struct S22 {};
template<typename T> using U22 = volatile S22<T>;
int f22(vector<volatile const U22<int>>);
int k22 = f22(vector<volatile U22<int>>());
// CHECK-ELIDE-NOTREE: no matching function for call to 'f22'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<U22<[...]>>' to 'vector<const U22<[...]>>' for 1st argument 
// CHECK-NOELIDE-NOTREE: no matching function for call to 'f22'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'vector<U22<int>>' to 'vector<const U22<int>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'f22'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:    vector<
// CHECK-ELIDE-TREE:      [(no qualifiers) != const] U22<
// CHECK-ELIDE-TREE:        [...]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'f22'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:    vector<
// CHECK-NOELIDE-TREE:      [(no qualifiers) != const] U22<
// CHECK-NOELIDE-TREE:        int>>

// Testing qualifiers and typedefs.
template <class T> struct D23{};
template <class T> using C23 = D23<T>;
typedef const C23<int> B23;
template<class ...T> using A23 = B23;

void foo23(D23<A23<>> b) {}
void test23() {
  foo23(D23<D23<char>>());
  foo23(C23<char>());
}

// CHECK-ELIDE-NOTREE: no matching function for call to 'foo23'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'D23<D23<char>>' to 'D23<const D23<int>>' for 1st argument
// CHECK-ELIDE-NOTREE: no matching function for call to 'foo23'
// CHECK-ELIDE-NOTREE: candidate function not viable: no known conversion from 'D23<char>' to 'D23<A23<>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'foo23'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'D23<D23<char>>' to 'D23<const D23<int>>' for 1st argument
// CHECK-NOELIDE-NOTREE: no matching function for call to 'foo23'
// CHECK-NOELIDE-NOTREE: candidate function not viable: no known conversion from 'D23<char>' to 'D23<A23<>>' for 1st argument
// CHECK-ELIDE-TREE: no matching function for call to 'foo23'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   D23<
// CHECK-ELIDE-TREE:     [(no qualifiers) != const] D23<
// CHECK-ELIDE-TREE:       [char != int]>>
// CHECK-ELIDE-TREE: no matching function for call to 'foo23'
// CHECK-ELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   D23<
// CHECK-ELIDE-TREE:     [char != A23<>]>
// CHECK-NOELIDE-TREE: no matching function for call to 'foo23'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   D23<
// CHECK-NOELIDE-TREE:     [(no qualifiers) != const] D23<
// CHECK-NOELIDE-TREE:       [char != int]>>
// CHECK-NOELIDE-TREE: no matching function for call to 'foo23'
// CHECK-NOELIDE-TREE: candidate function not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   D23<
// CHECK-NOELIDE-TREE:     [char != A23<>]>

namespace PR14015 {
template <unsigned N> class Foo1 {};
template <unsigned N = 2> class Foo2 {};
template <unsigned ...N> class Foo3 {};

void Play1() {
  Foo1<1> F1;
  Foo1<2> F2, F3;
  F2 = F1;
  F1 = F2;
  F2 = F3;
  F3 = F2;
}

// CHECK-ELIDE-NOTREE: no viable overloaded '='
// CHECK-ELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo1<1>' to 'const Foo1<2>' for 1st argument
// CHECK-ELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo1<1>' to 'Foo1<2>' for 1st argument
// CHECK-ELIDE-NOTREE: no viable overloaded '='
// CHECK-ELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo1<2>' to 'const Foo1<1>' for 1st argument
// CHECK-ELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo1<2>' to 'Foo1<1>' for 1st argument
// CHECK-NOELIDE-NOTREE: no viable overloaded '='
// CHECK-NOELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo1<1>' to 'const Foo1<2>' for 1st argument
// CHECK-NOELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo1<1>' to 'Foo1<2>' for 1st argument
// CHECK-NOELIDE-NOTREE: no viable overloaded '='
// CHECK-NOELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo1<2>' to 'const Foo1<1>' for 1st argument
// CHECK-NOELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo1<2>' to 'Foo1<1>' for 1st argument
// CHECK-ELIDE-TREE: no viable overloaded '='
// CHECK-ELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   [(no qualifiers) != const] Foo1<
// CHECK-ELIDE-TREE:     [1 != 2]>
// CHECK-ELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Foo1<
// CHECK-ELIDE-TREE:     [1 != 2]>
// CHECK-ELIDE-TREE: no viable overloaded '='
// CHECK-ELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   [(no qualifiers) != const] Foo1<
// CHECK-ELIDE-TREE:     [2 != 1]>
// CHECK-ELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Foo1<
// CHECK-ELIDE-TREE:     [2 != 1]>
// CHECK-NOELIDE-TREE: no viable overloaded '='
// CHECK-NOELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   [(no qualifiers) != const] Foo1<
// CHECK-NOELIDE-TREE:     [1 != 2]>
// CHECK-NOELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Foo1<
// CHECK-NOELIDE-TREE:     [1 != 2]>
// CHECK-NOELIDE-TREE: no viable overloaded '='
// CHECK-NOELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   [(no qualifiers) != const] Foo1<
// CHECK-NOELIDE-TREE:     [2 != 1]>
// CHECK-NOELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Foo1<
// CHECK-NOELIDE-TREE:     [2 != 1]>

void Play2() {
  Foo2<1> F1;
  Foo2<> F2, F3;
  F2 = F1;
  F1 = F2;
  F2 = F3;
  F3 = F2;
}
// CHECK-ELIDE-NOTREE: no viable overloaded '='
// CHECK-ELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo2<1>' to 'const Foo2<2>' for 1st argument
// CHECK-ELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo2<1>' to 'Foo2<2>' for 1st argument
// CHECK-ELIDE-NOTREE: no viable overloaded '='
// CHECK-ELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo2<(default) 2>' to 'const Foo2<1>' for 1st argument
// CHECK-ELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo2<(default) 2>' to 'Foo2<1>' for 1st argument
// CHECK-NOELIDE-NOTREE: no viable overloaded '='
// CHECK-NOELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo2<1>' to 'const Foo2<2>' for 1st argument
// CHECK-NOELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo2<1>' to 'Foo2<2>' for 1st argument
// CHECK-NOELIDE-NOTREE: no viable overloaded '='
// CHECK-NOELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo2<(default) 2>' to 'const Foo2<1>' for 1st argument
// CHECK-NOELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo2<(default) 2>' to 'Foo2<1>' for 1st argument
// CHECK-ELIDE-TREE: no viable overloaded '='
// CHECK-ELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   [(no qualifiers) != const] Foo2<
// CHECK-ELIDE-TREE:     [1 != 2]>
// CHECK-ELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Foo2<
// CHECK-ELIDE-TREE:     [1 != 2]>
// CHECK-ELIDE-TREE: no viable overloaded '='
// CHECK-ELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   [(no qualifiers) != const] Foo2<
// CHECK-ELIDE-TREE:     [(default) 2 != 1]>
// CHECK-ELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Foo2<
// CHECK-ELIDE-TREE:     [(default) 2 != 1]>
// CHECK-NOELIDE-TREE: no viable overloaded '='
// CHECK-NOELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   [(no qualifiers) != const] Foo2<
// CHECK-NOELIDE-TREE:     [1 != 2]>
// CHECK-NOELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Foo2<
// CHECK-NOELIDE-TREE:     [1 != 2]>
// CHECK-NOELIDE-TREE: no viable overloaded '='
// CHECK-NOELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   [(no qualifiers) != const] Foo2<
// CHECK-NOELIDE-TREE:     [(default) 2 != 1]>
// CHECK-NOELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Foo2<
// CHECK-NOELIDE-TREE:     [(default) 2 != 1]>

void Play3() {
  Foo3<1> F1;
  Foo3<2, 1> F2, F3;
  F2 = F1;
  F1 = F2;
  F2 = F3;
  F3 = F2;
}
// CHECK-ELIDE-NOTREE: no viable overloaded '='
// CHECK-ELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo3<1, (no argument)>' to 'const Foo3<2, 1>' for 1st argument
// CHECK-ELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo3<1, (no argument)>' to 'Foo3<2, 1>' for 1st argument
// CHECK-ELIDE-NOTREE: no viable overloaded '='
// CHECK-ELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo3<2, 1>' to 'const Foo3<1, (no argument)>' for 1st argument
// CHECK-ELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo3<2, 1>' to 'Foo3<1, (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: no viable overloaded '='
// CHECK-NOELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo3<1, (no argument)>' to 'const Foo3<2, 1>' for 1st argument
// CHECK-NOELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo3<1, (no argument)>' to 'Foo3<2, 1>' for 1st argument
// CHECK-NOELIDE-NOTREE: no viable overloaded '='
// CHECK-NOELIDE-NOTREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from 'Foo3<2, 1>' to 'const Foo3<1, (no argument)>' for 1st argument
// CHECK-NOELIDE-NOTREE: candidate function (the implicit move assignment operator) not viable: no known conversion from 'Foo3<2, 1>' to 'Foo3<1, (no argument)>' for 1st argument
// CHECK-ELIDE-TREE: no viable overloaded '='
// CHECK-ELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   [(no qualifiers) != const] Foo3<
// CHECK-ELIDE-TREE:     [1 != 2], 
// CHECK-ELIDE-TREE:     [(no argument) != 1]>
// CHECK-ELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Foo3<
// CHECK-ELIDE-TREE:     [1 != 2],
// CHECK-ELIDE-TREE:     [(no argument) != 1]>
// CHECK-ELIDE-TREE: no viable overloaded '='
// CHECK-ELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   [(no qualifiers) != const] Foo3<
// CHECK-ELIDE-TREE:     [2 != 1],
// CHECK-ELIDE-TREE:     [1 != (no argument)]>
// CHECK-ELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-ELIDE-TREE:   Foo3<
// CHECK-ELIDE-TREE:     [2 != 1], 
// CHECK-ELIDE-TREE:     [1 != (no argument)]>
// CHECK-NOELIDE-TREE: no viable overloaded '='
// CHECK-NOELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   [(no qualifiers) != const] Foo3<
// CHECK-NOELIDE-TREE:     [1 != 2], 
// CHECK-NOELIDE-TREE:     [(no argument) != 1]>
// CHECK-NOELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Foo3<
// CHECK-NOELIDE-TREE:     [1 != 2], 
// CHECK-NOELIDE-TREE:     [(no argument) != 1]>
// CHECK-NOELIDE-TREE: no viable overloaded '='
// CHECK-NOELIDE-TREE: candidate function (the implicit copy assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   [(no qualifiers) != const] Foo3<
// CHECK-NOELIDE-TREE:     [2 != 1], 
// CHECK-NOELIDE-TREE:     [1 != (no argument)]>
// CHECK-NOELIDE-TREE: candidate function (the implicit move assignment operator) not viable: no known conversion from argument type to parameter type for 1st argument
// CHECK-NOELIDE-TREE:   Foo3<
// CHECK-NOELIDE-TREE:     [2 != 1], 
// CHECK-NOELIDE-TREE:     [1 != (no argument)]>
}

namespace PR14342 {
  template<typename T, short a> struct X {};
  X<int, (signed char)-1> x = X<long, -1>();
  X<int, 3UL> y = X<int, 2>();
  // CHECK-ELIDE-NOTREE: error: no viable conversion from 'X<long, [...]>' to 'X<int, [...]>'
  // CHECK-ELIDE-NOTREE: error: no viable conversion from 'X<[...], 2>' to 'X<[...], 3>'
}

namespace PR14489 {
  // The important thing here is that the diagnostic diffs a template specialization
  // with no arguments against itself.  (We might need a different test if this
  // diagnostic changes).
  template<class ...V>
  struct VariableList   {
    void ConnectAllToAll(VariableList<>& params = VariableList<>())    {
    }
  };
  // CHECK-ELIDE-NOTREE: non-const lvalue reference to type 'VariableList<>' cannot bind to a temporary of type 'VariableList<>'
}

namespace rdar12456626 {
  struct IntWrapper {
    typedef int type;
  };
  
  template<typename T, typename T::type V>
  struct X { };
  
  struct A {
    virtual X<IntWrapper, 1> foo();
  };
  
  struct B : A {
    // CHECK-ELIDE-NOTREE: virtual function 'foo' has a different return type
    virtual X<IntWrapper, 2> foo();
  };
}

namespace PR15023 {
  // Don't crash when non-QualTypes are passed to a diff modifier.
  template <typename... Args>
  void func(void (*func)(Args...), Args...) { }

  void bar(int, int &) {
  }

  void foo(int x) {
    func(bar, 1, x)
  }
  // CHECK-ELIDE-NOTREE: no matching function for call to 'func'
  // CHECK-ELIDE-NOTREE: candidate template ignored: deduced conflicting types for parameter 'Args' (<int, int &> vs. <int, int>)
}

namespace rdar12931988 {
  namespace A {
    template<typename T> struct X { };
  }

  namespace B {
    template<typename T> struct X { };
  }

  void foo(A::X<int> &ax, B::X<int> bx) {
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'B::X<int>' to 'const rdar12931988::A::X<int>'
    ax = bx;
  }

  template<template<typename> class> class Y {};

  void bar(Y<A::X> ya, Y<B::X> yb) {
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'Y<template rdar12931988::B::X>' to 'Y<template rdar12931988::A::X>'
    ya = yb;
  }
}

namespace ValueDecl {
  int int1, int2, default_int;
  template <const int& T = default_int>
  struct S {};

  typedef S<int1> T1;
  typedef S<int2> T2;
  typedef S<> TD;

  void test() {
    T1 t1;
    T2 t2;
    TD td;

    t1 = t2;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'S<int2>' to 'S<int1>'

    t2 = t1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'S<int1>' to 'S<int2>'

    td = t1;
    // TODO: Find out why (default) isn't printed on second template.
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'S<int1>' to 'S<default_int>'

    t2 = td;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'S<(default) default_int>' to 'S<int2>'

  }
}

namespace DependentDefault {
  template <typename> struct Trait {
    enum { V = 40 };
    typedef int Ty;
    static int I;
  };
  int other;

  template <typename T, int = Trait<T>::V > struct A {};
  template <typename T, typename = Trait<T>::Ty > struct B {};
  template <typename T, int& = Trait<T>::I > struct C {};

  void test() {

    A<int> a1;
    A<char> a2;
    A<int, 10> a3;
    a1 = a2;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'A<char, [...]>' to 'A<int, [...]>'
    a3 = a1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'A<[...], (default) 40>' to 'A<[...], 10>'
    a2 = a3;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'A<int, 10>' to 'A<char, 40>'

    B<int> b1;
    B<char> b2;
    B<int, char> b3;
    b1 = b2;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'B<char, [...]>' to 'B<int, [...]>'
    b3 = b1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'B<[...], (default) int>' to 'B<[...], char>'
    b2 = b3;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'B<int, char>' to 'B<char, int>'

    C<int> c1;
    C<char> c2;
    C<int, other> c3;
    c1 = c2;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'C<char, (default) I>' to 'C<int, I>'
    c3 = c1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'C<[...], (default) I>' to 'C<[...], other>'
    c2 = c3;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'C<int, other>' to 'C<char, I>'
  }
}

namespace VariadicDefault {
  int i1, i2, i3;
  template <int = 5, int...> struct A {};
  template <int& = i1, int& ...> struct B {};
  template <typename = void, typename...> struct C {};

  void test() {
    A<> a1;
    A<5, 6, 7> a2;
    A<1, 2> a3;
    a2 = a1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'A<[...], (no argument), (no argument)>' to 'A<[...], 6, 7>'
    a3 = a1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'A<(default) 5, (no argument)>' to 'A<1, 2>'

    B<> b1;
    B<i1, i2, i3> b2;
    B<i2, i3> b3;
    b2 = b1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'B<[...], (no argument), (no argument)>' to 'B<[...], i2, i3>'
    b3 = b1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'B<(default) i1, (no argument)>' to 'B<i2, i3>'

    B<i1, i2, i3> b4 = b1;
    // CHECK-ELIDE-NOTREE: no viable conversion from 'B<[...], (no argument), (no argument)>' to 'B<[...], i2, i3>'
    B<i2, i3> b5 = b1;
    // CHECK-ELIDE-NOTREE: no viable conversion from 'B<(default) i1, (no argument)>' to 'B<i2, i3>'

    C<> c1;
    C<void, void> c2;
    C<char, char> c3;
    c2 = c1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'C<[...], (no argument)>' to 'C<[...], void>'
    c3 = c1;
    // CHECK-ELIDE-NOTREE: no viable overloaded '='
    // CHECK-ELIDE-NOTREE: no known conversion from 'C<(default) void, (no argument)>' to 'C<char, char>'
  }
}

namespace PointerArguments {
  template <int *p> class T {};
  template <int* ...> class U {};
  int a, b, c;
  int z[5];
  void test() {
    T<&a> ta;
    T<z> tz;
    T<&b> tb(ta);
    // CHECK-ELIDE-NOTREE: no matching constructor for initialization of 'T<&b>'
    // CHECK-ELIDE-NOTREE: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'T<&a>' to 'const T<&b>' for 1st argument
    T<&c> tc(tz);
    // CHECK-ELIDE-NOTREE: no matching constructor for initialization of 'T<&c>'
    // CHECK-ELIDE-NOTREE: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'T<z>' to 'const T<&c>' for 1st argument

    U<&a, &a> uaa;
    U<&b> ub(uaa);
    // CHECK-ELIDE-NOTREE: no matching constructor for initialization of 'U<&b>'
    // CHECK-ELIDE-NOTREE: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'U<&a, &a>' to 'const U<&b, (no argument)>' for 1st argument

    U<&b, &b, &b> ubbb(uaa);
    // CHECK-ELIDE-NOTREE: no matching constructor for initialization of 'U<&b, &b, &b>'
    // CHECK-ELIDE-NOTREE: candidate constructor (the implicit copy constructor) not viable: no known conversion from 'U<&a, &a, (no argument)>' to 'const U<&b, &b, &b>' for 1st argument

  }
}

namespace DependentInt {
  template<int Num> struct INT;

  template <class CLASS, class Int_wrapper = INT<CLASS::val> >
  struct C;

  struct N {
    static const int val = 1;
  };

  template <class M_T>
  struct M {};

  void test() {
    using T1 = M<C<int, INT<0>>>;
    using T2 = M<C<N>>;
    T2 p;
    T1 x = p;
    // CHECK-ELIDE-NOTREE: no viable conversion from 'M<C<DependentInt::N, INT<1>>>' to 'M<C<int, INT<0>>>'
  }
}

namespace PR17510 {
class Atom;

template <typename T> class allocator;
template <typename T, typename A> class vector;

typedef vector<const Atom *, allocator<const Atom *> > AtomVector;

template <typename T, typename A = allocator<const Atom *> > class vector {};

void foo() {
  vector<Atom *> v;
  AtomVector v2(v);
  // CHECK-ELIDE-NOTREE: no known conversion from 'vector<PR17510::Atom *, [...]>' to 'const vector<const PR17510::Atom *, [...]>'
}
}

namespace PR15677 {
template <bool>
struct A{};

template <typename T>
using B = A<T::value>;

template <typename T>
using B = A<!T::value>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('A<!T::value>' vs 'A<T::value>')

template <int>
struct C{};

template <typename T>
using D = C<T::value>;

template <typename T>
using D = C<T::value + 1>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('C<T::value + 1>' vs 'C<T::value>')

template <typename T>
using E = C<T::value>;

template <typename T>
using E = C<42>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('C<42>' vs 'C<T::value>')

template <typename T>
using F = C<T::value>;

template <typename T>
using F = C<21 + 21>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('C<21 + 21 aka 42>' vs 'C<T::value>')
}
}

namespace AddressOf {
template <int*>
struct S {};

template <class T>
struct Wrapper {};

template <class T>
Wrapper<T> MakeWrapper();
int global, global2;
constexpr int * ptr = nullptr;
Wrapper<S<ptr>> W = MakeWrapper<S<&global>>();
// Don't print an extra '&' for 'ptr'
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global>>' to 'Wrapper<S<ptr>>'

// Handle parens correctly
Wrapper<S<(&global2)>> W2 = MakeWrapper<S<&global>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global>>' to 'Wrapper<S<&global2>>'
Wrapper<S<&global2>> W3 = MakeWrapper<S<(&global)>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global>>' to 'Wrapper<S<&global2>>'
Wrapper<S<(&global2)>> W4 = MakeWrapper<S<(&global)>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global>>' to 'Wrapper<S<&global2>>'
}

namespace NullPtr {
template <int*, int*>
struct S {};

template <class T>
struct Wrapper {};

template <class T>
Wrapper<T> MakeWrapper();
int global, global2;
constexpr int * ptr = nullptr;
constexpr int * ptr2 = static_cast<int*>(0);

S<&global> s1 = S<&global, ptr>();
S<&global, nullptr> s2 = S<&global, ptr>();

S<&global, nullptr> s3 = S<&global, &global>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'S<[...], &global>' to 'S<[...], nullptr>'
S<&global, ptr> s4 = S<&global, &global>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'S<[...], &global>' to 'S<[...], ptr>

Wrapper<S<&global, nullptr>> W1 = MakeWrapper<S<&global, ptr>>();
Wrapper<S<&global, static_cast<int*>(0)>> W2 = MakeWrapper<S<&global, ptr>>();

Wrapper<S<&global, nullptr>> W3 = MakeWrapper<S<&global, &global>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<[...], &global>>' to 'Wrapper<S<[...], nullptr>>'
Wrapper<S<&global, ptr>> W4 = MakeWrapper<S<&global, &global>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<[...], &global>>' to 'Wrapper<S<[...], ptr>>'

Wrapper<S<&global2, ptr>> W5 = MakeWrapper<S<&global, nullptr>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global, [...]>>' to 'Wrapper<S<&global2, [...]>>'
Wrapper<S<&global2, nullptr>> W6 = MakeWrapper<S<&global, nullptr>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global, [...]>>' to 'Wrapper<S<&global2, [...]>>'
Wrapper<S<&global2, ptr2>> W7 = MakeWrapper<S<&global, nullptr>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global, [...]>>' to 'Wrapper<S<&global2, [...]>>'
Wrapper<S<&global2, nullptr>> W8 = MakeWrapper<S<&global, ptr2>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global, [...]>>' to 'Wrapper<S<&global2, [...]>>'
Wrapper<S<&global2, ptr>> W9 = MakeWrapper<S<&global, ptr2>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global, [...]>>' to 'Wrapper<S<&global2, [...]>>'
Wrapper<S<&global2, ptr2>> W10 = MakeWrapper<S<&global, ptr>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global, [...]>>' to 'Wrapper<S<&global2, [...]>>'
Wrapper<S<&global2, static_cast<int *>(0)>> W11 =
    MakeWrapper<S<&global, nullptr>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global, [...]>>' to 'Wrapper<S<&global2, [...]>>'
Wrapper<S<&global2, nullptr>> W12 =
    MakeWrapper<S<&global, static_cast<int *>(0)>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<&global, [...]>>' to 'Wrapper<S<&global2, [...]>>'

Wrapper<S<&global, &global>> W13 = MakeWrapper<S<&global, ptr>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<[...], nullptr>>' to 'Wrapper<S<[...], &global>>'
Wrapper<S<&global, ptr>> W14 = MakeWrapper<S<&global, &global>>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'Wrapper<S<[...], &global>>' to 'Wrapper<S<[...], ptr>>'
}

namespace TemplateTemplateDefault {
template <class> class A{};
template <class> class B{};
template <class> class C{};
template <template <class> class, template <class> class = A>
        class T {};

T<A> t1 = T<A, C>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'T<[...], template C>' to 'T<[...], (default) template A>'
T<A, C> t2 = T<A>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'T<[...], (default) template A>' to 'T<[...], template C>'
T<A> t3 = T<B>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'T<template B>' to 'T<template A>'
T<B, C> t4 = T<C, B>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'T<template C, template B>' to 'T<template B, template C>'
T<A, A> t5 = T<B>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'T<template B, [...]>' to 'T<template A, [...]>'
T<B> t6 = T<A, A>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'T<template A, [...]>' to 'T<template B, [...]>'
}

namespace Bool {
template <class> class A{};
A<bool> a1 = A<int>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'A<int>' to 'A<bool>'
A<int> a2 = A<bool>();
// CHECK-ELIDE-NOTREE: no viable conversion from 'A<bool>' to 'A<int>'
}

namespace TypeAlias {
template <int, int = 0> class A {};

template <class T> using a = A<T::num, 0>;
template <class T> using a = A<T::num>;

template <class T> using A1 = A<T::num>;
template <class T> using A1 = A<T::num + 0>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('A<T::num + 0>' vs 'A<T::num>')

template <class T> using A2 = A<1 + T::num>;
template <class T> using A2 = A<T::num + 1>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('A<T::num + 1>' vs 'A<1 + T::num>')

template <class T> using A3 = A<(T::num)>;
template <class T> using A3 = A<T::num>;
// CHECK-ELIDE-NOTREE: error: type alias template redefinition with different types ('A<T::num>' vs 'A<(T::num)>')

          template <class T> using A4 = A<(T::num)>;
template <class T> using A4 = A<((T::num))>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('A<((T::num))>' vs 'A<(T::num)>')

template <class T> using A5 = A<T::num, 1>;
template <class T> using A5 = A<T::num>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('A<[...], (default) 0>' vs 'A<[...], 1>')

template <class T> using A6 = A<T::num + 5, 1>;
template <class T> using A6 = A<T::num + 5>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('A<[...], (default) 0>' vs 'A<[...], 1>')

template <class T> using A7 = A<T::num, 1>;
template <class T> using A7 = A<(T::num)>;
// CHECK-ELIDE-NOTREE: type alias template redefinition with different types ('A<(T::num), (default) 0>' vs 'A<T::num, 1>')
}

namespace TemplateArgumentImplicitConversion {
template <int X> struct condition {};

struct is_const {
    constexpr operator int() const { return 10; }
};

using T = condition<(is_const())>;
void foo(const T &t) {
  T &t2 = t;
}
// CHECK-ELIDE-NOTREE: binding of reference to type 'condition<[...]>' to a value of type 'const condition<[...]>' drops qualifiers
}

// CHECK-ELIDE-NOTREE: {{[0-9]*}} errors generated.
// CHECK-NOELIDE-NOTREE: {{[0-9]*}} errors generated.
// CHECK-ELIDE-TREE: {{[0-9]*}} errors generated.
// CHECK-NOELIDE-TREE: {{[0-9]*}} errors generated.

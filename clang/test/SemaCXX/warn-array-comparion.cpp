// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s -verify=expected,not-cxx20
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wno-deprecated-array-compare -verify %s -verify=expected,not-cxx20
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wdeprecated -verify %s -verify=expected,cxx20
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wdeprecated -verify %s -verify=expected,cxx26

typedef struct {
  char str[16];
  int id[16];
} Object;

bool object_equal(const Object &obj1, const Object &obj2) {
  if (obj1.str != obj2.str) // not-cxx20-warning {{comparison between two arrays compare their addresses}} cxx20-warning {{comparison between two arrays is deprecated}}
    return false;           // cxx26-error@-1 {{comparison between two arrays is ill-formed in C++26}}
  if (obj1.id != obj2.id) // not-cxx20-warning {{comparison between two arrays compare their addresses}} cxx20-warning {{comparison between two arrays is deprecated}}
    return false;         // cxx26-error@-1 {{comparison between two arrays is ill-formed in C++26}}
  return true;
}


void foo(int (&array1)[2], int (&array2)[2]) {
  if (array1 == array2) { } // not-cxx20-warning {{comparison between two arrays compare their addresses}} cxx20-warning {{comparison between two arrays is deprecated}}
                            // cxx26-error@-1 {{comparison between two arrays is ill-formed in C++26}}
}

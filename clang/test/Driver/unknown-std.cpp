// This file checks output given when processing C++/ObjC++ files.
// When user selects invalid language standard
// print out supported values with short description.

// RUN: not %clang %s -std=foobar -c 2>&1 | FileCheck --match-full-lines %s
// RUN: not %clang -x objective-c++ %s -std=foobar -c 2>&1 | FileCheck --match-full-lines %s
// RUN: not %clang -x cuda -nocudainc -nocudalib --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:   %s -std=foobar -c 2>&1 | FileCheck --match-full-lines %s
// RUN: not %clang -x hip -nocudainc -nocudalib %s -std=foobar -c 2>&1 \
// RUN:   | FileCheck --match-full-lines %s

// CHECK: error: invalid value 'foobar' in '-std=foobar'
// CHECK-NEXT: note: use 'c++98' or 'c++03' for 'ISO C++ 1998 with amendments' standard
// CHECK-NEXT: note: use 'gnu++98' or 'gnu++03' for 'ISO C++ 1998 with amendments and GNU extensions' standard
// CHECK-NEXT: note: use 'c++11' for 'ISO C++ 2011 with amendments' standard
// CHECK-NEXT: note: use 'gnu++11' for 'ISO C++ 2011 with amendments and GNU extensions' standard
// CHECK-NEXT: note: use 'c++14' for 'ISO C++ 2014 with amendments' standard
// CHECK-NEXT: note: use 'gnu++14' for 'ISO C++ 2014 with amendments and GNU extensions' standard
// CHECK-NEXT: note: use 'c++17' for 'ISO C++ 2017 with amendments' standard
// CHECK-NEXT: note: use 'gnu++17' for 'ISO C++ 2017 with amendments and GNU extensions' standard
// CHECK-NEXT: note: use 'c++20' for 'ISO C++ 2020 DIS' standard
// CHECK-NEXT: note: use 'gnu++20' for 'ISO C++ 2020 DIS with GNU extensions' standard
// CHECK-NEXT: note: use 'c++23' for 'ISO C++ 2023 DIS' standard
// CHECK-NEXT: note: use 'gnu++23' for 'ISO C++ 2023 DIS with GNU extensions' standard
// CHECK-NEXT: note: use 'c++2c' or 'c++26' for 'Working draft for C++2c' standard
// CHECK-NEXT: note: use 'gnu++2c' or 'gnu++26' for 'Working draft for C++2c with GNU extensions' standard

// Make sure that no other output is present.
// CHECK-NOT: {{^.+$}}


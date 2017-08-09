// This file checks output given when processing C++/ObjC++ files.
// When user selects invalid language standard
// print out supported values with short description.

// RUN: not %clang %s -std=foobar -c 2>&1 | FileCheck --match-full-lines %s
// RUN: not %clang -x objective-c++ %s -std=foobar -c 2>&1 | FileCheck --match-full-lines %s
// RUN: not %clang -x cuda -nocudainc -nocudalib %s -std=foobar -c 2>&1 | FileCheck --match-full-lines --check-prefix=CHECK --check-prefix=CUDA %s

// CHECK: error: invalid value 'foobar' in '-std=foobar'
// CHECK-NEXT: note: use 'c++98' or 'c++03' for 'ISO C++ 1998 with amendments' standard
// CHECK-NEXT: note: use 'gnu++98' or 'gnu++03' for 'ISO C++ 1998 with amendments and GNU extensions' standard
// CHECK-NEXT: note: use 'c++11' for 'ISO C++ 2011 with amendments' standard
// CHECK-NEXT: note: use 'gnu++11' for 'ISO C++ 2011 with amendments and GNU extensions' standard
// CHECK-NEXT: note: use 'c++14' for 'ISO C++ 2014 with amendments' standard
// CHECK-NEXT: note: use 'gnu++14' for 'ISO C++ 2014 with amendments and GNU extensions' standard
// CHECK-NEXT: note: use 'c++17' for 'ISO C++ 2017 with amendments' standard
// CHECK-NEXT: note: use 'gnu++17' for 'ISO C++ 2017 with amendments and GNU extensions' standard
// CHECK-NEXT: note: use 'c++2a' for 'Working draft for ISO C++ 2020' standard
// CHECK-NEXT: note: use 'gnu++2a' for 'Working draft for ISO C++ 2020 with GNU extensions' standard
// CUDA-NEXT: note: use 'cuda' for 'NVIDIA CUDA(tm)' standard

// Make sure that no other output is present.
// CHECK-NOT: {{^.+$}}


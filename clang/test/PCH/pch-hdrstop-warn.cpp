// Create PCH with #pragma hdrstop
// RUN: %clang_cc1 -verify -I %S -emit-pch -pch-through-hdrstop-create \
// RUN:   -fms-extensions -o %t.pch -x c++-header %s

// Create PCH object with #pragma hdrstop
// RUN: %clang_cc1 -verify -I %S -emit-obj -include-pch %t.pch \
// RUN:   -pch-through-hdrstop-create -fms-extensions -o %t.obj -x c++ %s

//expected-warning@+1{{hdrstop filename not supported}}
#pragma hdrstop("name.pch")

// RUN: mkdir -p %T/move-type-alias
// RUN: cp %S/Inputs/type_alias.h  %T/move-type-alias/type_alias.h
// RUN: echo '#include "type_alias.h"' > %T/move-type-alias/type_alias.cpp
// RUN: cd %T/move-type-alias
//
// -----------------------------------------------------------------------------
// Test moving typedef declarations.
// -----------------------------------------------------------------------------
// RUN: clang-move -names="Int1" -new_cc=%T/move-type-alias/new_test.cpp -new_header=%T/move-type-alias/new_test.h -old_cc=%T/move-type-alias/type_alias.cpp -old_header=%T/move-type-alias/type_alias.h %T/move-type-alias/type_alias.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/move-type-alias/new_test.h -check-prefix=CHECK-NEW-TEST-H-CASE1 %s
// RUN: FileCheck -input-file=%T/move-type-alias/type_alias.h -check-prefix=CHECK-OLD-TEST-H-CASE1 %s

// CHECK-NEW-TEST-H-CASE1: typedef int Int1;

// CHECK-OLD-TEST-H-CASE1-NOT: typedef int Int1;


// -----------------------------------------------------------------------------
// Test moving type alias declarations.
// -----------------------------------------------------------------------------
// RUN: cp %S/Inputs/type_alias.h  %T/move-type-alias/type_alias.h
// RUN: echo '#include "type_alias.h"' > %T/move-type-alias/type_alias.cpp
// RUN: clang-move -names="Int2" -new_cc=%T/move-type-alias/new_test.cpp -new_header=%T/move-type-alias/new_test.h -old_cc=%T/move-type-alias/type_alias.cpp -old_header=%T/move-type-alias/type_alias.h %T/move-type-alias/type_alias.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/move-type-alias/new_test.h -check-prefix=CHECK-NEW-TEST-H-CASE2 %s
// RUN: FileCheck -input-file=%T/move-type-alias/type_alias.h -check-prefix=CHECK-OLD-TEST-H-CASE2 %s

// CHECK-NEW-TEST-H-CASE2: using Int2 = int;

// CHECK-OLD-TEST-H-CASE2-NOT: using Int2 = int;


// -----------------------------------------------------------------------------
// Test moving template type alias declarations.
// -----------------------------------------------------------------------------
// RUN: cp %S/Inputs/type_alias.h  %T/move-type-alias/type_alias.h
// RUN: echo '#include "type_alias.h"' > %T/move-type-alias/type_alias.cpp
// RUN: clang-move -names="B" -new_cc=%T/move-type-alias/new_test.cpp -new_header=%T/move-type-alias/new_test.h -old_cc=%T/move-type-alias/type_alias.cpp -old_header=%T/move-type-alias/type_alias.h %T/move-type-alias/type_alias.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/move-type-alias/new_test.h -check-prefix=CHECK-OLD-TEST-H-CASE3 %s

// CHECK-NEW-TEST-H-CASE3: template<class T> using B = A<T>;
// CHECK-OLD-TEST-H-CASE3-NOT: template<class T> using B = A<T>;


// -----------------------------------------------------------------------------
// Test not moving class-insided typedef declarations.
// -----------------------------------------------------------------------------
// RUN: cp %S/Inputs/type_alias.h  %T/move-type-alias/type_alias.h
// RUN: echo '#include "type_alias.h"' > %T/move-type-alias/type_alias.cpp
// RUN: clang-move -names="C::Int3" -new_cc=%T/move-type-alias/new_test.cpp -new_header=%T/move-type-alias/new_test.h -old_cc=%T/move-type-alias/type_alias.cpp -old_header=%T/move-type-alias/type_alias.h %T/move-type-alias/type_alias.cpp -- -std=c++11
// RUN: FileCheck -input-file=%T/move-type-alias/new_test.h -allow-empty -check-prefix=CHECK-EMPTY %s

// CHECK-EMPTY: {{^}}{{$}}

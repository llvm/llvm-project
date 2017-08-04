// RUN: mkdir -p %T/move-var
// RUN: cp %S/Inputs/var_test*  %T/move-var
// RUN: cd %T/move-var
// RUN: clang-move -names="a::kGlobalInt" -new_header=%T/move-var/new_var_test.h -old_header=../move-var/var_test.h -old_cc=../move-var/var_test.cpp -new_cc=%T/move-var/new_var_test.cpp %T/move-var/var_test.cpp --
// RUN: FileCheck -input-file=%T/move-var/var_test.h -check-prefix=CHECK-OLD-VAR-H-CASE1 %s
// RUN: FileCheck -input-file=%T/move-var/var_test.cpp -check-prefix=CHECK-OLD-VAR-CPP-CASE1 %s
// RUN: FileCheck -input-file=%T/move-var/new_var_test.h -check-prefix=CHECK-NEW-VAR-H-CASE1 %s
// RUN: FileCheck -input-file=%T/move-var/new_var_test.cpp -check-prefix=CHECK-NEW-VAR-CPP-CASE1 %s

// CHECK-OLD-VAR-H-CASE1-NOT: extern int kGlobalInt;
// CHECK-OLD-VAR-H-CASE1: int kGlobalInt = 3;

// CHECK-OLD-VAR-CPP-CASE1-NOT: int kGlobalInt = 1;

// CHECK-NEW-VAR-H-CASE1: extern int kGlobalInt;
// CHECK-NEW-VAR-H-CASE1-NOT: int kGlobalInt = 3;

// CHECK-NEW-VAR-CPP-CASE1: int kGlobalInt = 1;


// RUN: cp %S/Inputs/var_test*  %T/move-var
// RUN: clang-move -names="a::kGlobalStr" -new_header=%T/move-var/new_var_test.h -old_header=../move-var/var_test.h -old_cc=../move-var/var_test.cpp -new_cc=%T/move-var/new_var_test.cpp %T/move-var/var_test.cpp --
// RUN: FileCheck -input-file=%T/move-var/var_test.h -check-prefix=CHECK-OLD-VAR-H-CASE2 %s
// RUN: FileCheck -input-file=%T/move-var/var_test.cpp -check-prefix=CHECK-OLD-VAR-CPP-CASE2 %s
// RUN: FileCheck -input-file=%T/move-var/new_var_test.h -check-prefix=CHECK-NEW-VAR-H-CASE2 %s
// RUN: FileCheck -input-file=%T/move-var/new_var_test.cpp -check-prefix=CHECK-NEW-VAR-CPP-CASE2 %s

// CHECK-OLD-VAR-H-CASE2-NOT: extern const char *const kGlobalStr;
// CHECK-OLD-VAR-H-CASE2: const char *const kGlobalStr = "Hello2";

// CHECK-OLD-VAR-CPP-CASE2-NOT: const char *const kGlobalStr = "Hello";

// CHECK-NEW-VAR-H-CASE2: extern const char *const kGlobalStr;
// CHECK-NEW-VAR-H-CASE2-NOT: const char *const kGlobalStr = "Hello2";

// CHECK-NEW-VAR-CPP-CASE2: const char *const kGlobalStr = "Hello";


// RUN: cp %S/Inputs/var_test*  %T/move-var
// RUN: clang-move -names="kEvilInt" -new_header=%T/move-var/new_var_test.h -old_header=../move-var/var_test.h -old_cc=../move-var/var_test.cpp -new_cc=%T/move-var/new_var_test.cpp %T/move-var/var_test.cpp --
// RUN: FileCheck -input-file=%T/move-var/var_test.h -check-prefix=CHECK-OLD-VAR-H-CASE3 %s
// RUN: FileCheck -input-file=%T/move-var/new_var_test.h -check-prefix=CHECK-NEW-VAR-H-CASE3 %s

// CHECK-OLD-VAR-H-CASE3-NOT: int kEvilInt = 2;

// CHECK-NEW-VAR-H-CASE3: int kEvilInt = 2;

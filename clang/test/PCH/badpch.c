// RUN: not %clang_cc1 -fsyntax-only -include-pch %S/Inputs/badpch-empty.h.gch %s 2>&1 | FileCheck -check-prefix=CHECK-EMPTY %s
// RUN: not %clang_cc1 -fsyntax-only -include-pch %S/Inputs/badpch-dir.h.gch %s 2>&1 | FileCheck -check-prefix=CHECK-DIR %s

// The purpose of this test is to verify that various invalid PCH files are
// reported as such.

// PR4568: The messages were much improved since the bug was filed on
// 2009-07-16, though in the case of the PCH being a directory, the error
// message still did not contain the name of the PCH. Also, r149918 which was
// submitted on 2012-02-06 introduced a segfault in the case where the PCH is
// an empty file and clang was built with assertions.
// CHECK-EMPTY: error: input is not a PCH file: '{{.*[/\\]}}badpch-empty.h.gch'
// CHECK-DIR:error: no suitable precompiled header file found in directory '{{.*[/\\]}}badpch-dir.h.gch

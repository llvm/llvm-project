// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -imacros %t/macros.h \
// RUN:   -include %t/first.h \
// RUN:   -include %t/second.h %t/M.cppm -verify
// RUN: %clang_cc1 -std=c++20 -x cuda -fsyntax-only -imacros %t/macros.h \
// RUN:   -include %t/first.h \
// RUN:   -include %t/second.h %t/M.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -imacros %t/macros.h \
// RUN:   -include %t/first.h \
// RUN:   -include %t/second.h %t/NoGMF.cppm -verify
// RUN: %clang_cc1 -std=c++20 -x cuda -fsyntax-only -imacros %t/macros.h \
// RUN:   -include %t/first.h \
// RUN:   -include %t/second.h %t/NoGMF.cppm -verify
// RUN: %clang_cc1 -std=c++20 -E -imacros %t/macros.h -include %t/first.h \
// RUN:   -include %t/second.h %t/M.cppm -o %t/M.ii
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -x c++-cpp-output %t/M.ii
// RUN: %clang_cc1 -std=c++20 -E -imacros %t/macros.h -include %t/first.h \
// RUN:   -include %t/second.h %t/NoGMF.cppm -o %t/NoGMF.ii
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -x c++-cpp-output %t/NoGMF.ii
// RUN: %clang_cc1 -std=c++20 -E -imacros %t/macros.h %t/MacroOnly.cppm \
// RUN:   | FileCheck %s --check-prefix=MACRO-ONLY
// RUN: %clang_cc1 -std=c++20 -E -include %t/Header.h %t/Preprocess.cppm \
// RUN:   | FileCheck %s --check-prefix=PREPROCESS
// RUN: %clang_cc1 -std=c++20 -x c++-header -emit-pch %t/pch.h -o %t/pch.pch
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -imacros %t/macros.h \
// RUN:   -include-pch %t/pch.pch -include %t/first.h %t/PCHGMF.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -imacros %t/macros.h \
// RUN:   -include-pch %t/pch.pch -include %t/first.h %t/PCHNoGMF.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -include-pch %t/pch.pch \
// RUN:   %t/PCHOnlyGMF.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -include-pch %t/pch.pch \
// RUN:   %t/PCHOnlyNoGMF.cppm -verify
// RUN: %clang_cc1 -std=c++20 -E -imacros %t/macros.h \
// RUN:   -include-pch %t/pch.pch -include %t/first.h %t/PCHNoGMF.cppm \
// RUN:   | FileCheck %s --check-prefix=PCH-ONLY
// RUN: %clang_cc1 -std=c++20 -E -include-pch %t/pch.pch \
// RUN:   %t/PCHOnlyNoGMF.cppm \
// RUN:   | FileCheck %s --check-prefix=PCH-ONLY-NO-GMF
// RUN: %clang_cc1 -std=c++20 -x cuda -emit-pch %t/pch.h -o %t/cuda.pch
// RUN: %clang_cc1 -std=c++20 -x cuda -fsyntax-only -imacros %t/macros.h \
// RUN:   -include-pch %t/cuda.pch -include %t/first.h %t/PCHGMF.cppm -verify
// RUN: %clang_cc1 -std=c++20 -x cuda -fsyntax-only -imacros %t/macros.h \
// RUN:   -include-pch %t/cuda.pch -include %t/first.h %t/PCHNoGMF.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -imacros %t/macros.h \
// RUN:   -include-pch %t/pch.pch -include %t/first.h %t/pch-tu.cpp -verify
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Base.cppm \
// RUN:   -o %t/Base.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -imacros %t/macros.h \
// RUN:   -include %t/first.h -include %t/second.h \
// RUN:   -fmodule-file=Base=%t/Base.pcm %t/Base-impl.cpp -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -imacros %t/macros.h \
// RUN:   -include %t/first.h \
// RUN:   -include %t/second.h %t/tu.cpp -verify

// MACRO-ONLY: __preprocessed_module{{ *}};
// PREPROCESS:      # 1 "<implicit-global-module-fragment>" 1
// PREPROCESS-NEXT: # 1 "<gmf-command-line-inputs>" 1
// PREPROCESS-NEXT: __preprocessed_module;
// PREPROCESS-NEXT: # 1 "{{.*}}Header.h" 1
// PREPROCESS-NEXT: struct Lexer {};
// PREPROCESS-NEXT: # 2 "<gmf-command-line-inputs>" 2
// PREPROCESS-NEXT: # 2 "<implicit-global-module-fragment>" 2
// PREPROCESS-NEXT: # 1 "{{.*}}Preprocess.cppm" 2
// PREPROCESS-NEXT: export __preprocessed_module M;
// PCH-ONLY: __preprocessed_module{{ *}};
// PCH-ONLY-NO-GMF: __preprocessed_module;
// PCH-ONLY-NO-GMF: export __preprocessed_module PCHOnlyNoGMF;

//--- macros.h
#define IMPLICIT_MACRO 3

//--- first.h
#define FIRST 1
static_assert(IMPLICIT_MACRO == 3);
struct FromFirst {};

//--- second.h
static_assert(FIRST == 1);
#define SECOND 2

//--- M.cppm
// expected-no-diagnostics
/* A leading comment and an escaped newline exercise raw-token detection. */
module \
;
static_assert(SECOND == 2);
export module M;
export FromFirst from_first();

//--- NoGMF.cppm
// expected-no-diagnostics
export module NoGMF;
static_assert(SECOND == 2);
export FromFirst no_gmf();

//--- MacroOnly.cppm
export module MacroOnly;
static_assert(IMPLICIT_MACRO == 3);

//--- Header.h
struct Lexer {};

//--- Preprocess.cppm
export module M;
export int count = 0;

//--- pch.h
#pragma once
struct FromPCH {};

//--- PCHGMF.cppm
// expected-no-diagnostics
module;
static_assert(IMPLICIT_MACRO == 3);
export module PCHGMF;
export FromPCH from_pch_gmf();
export FromFirst from_first_pch_gmf();

//--- PCHNoGMF.cppm
// expected-no-diagnostics
export module PCHNoGMF;
static_assert(IMPLICIT_MACRO == 3);
export FromPCH from_pch_no_gmf();
export FromFirst from_first_pch_no_gmf();

//--- PCHOnlyGMF.cppm
// expected-no-diagnostics
module;
export module PCHOnlyGMF;
export FromPCH from_pch_only_gmf();

//--- PCHOnlyNoGMF.cppm
// expected-no-diagnostics
export module PCHOnlyNoGMF;
export FromPCH from_pch_only_no_gmf();

//--- pch-tu.cpp
// expected-no-diagnostics
static_assert(IMPLICIT_MACRO == 3);
FromPCH from_pch_tu;
FromFirst from_first_pch_tu;

//--- Base.cppm
export module Base;
export void base();

//--- Base-impl.cpp
// expected-no-diagnostics
module Base;
static_assert(SECOND == 2);
FromFirst from_impl;

//--- tu.cpp
// expected-no-diagnostics
static_assert(FIRST == 1);
static_assert(SECOND == 2);
FromFirst from_first;

// RUN: rm -rf %t
// RUN: split-file %s %t

// Normal compilation for baseline.
// RUN: %clang_cc1 -x c-header %t/prefix.h -DSOME_MACRO=1 -emit-pch -o %t/prefix1.pch -Werror
// RUN: %clang_cc1 %t/t1.c -include-pch %t/prefix1.pch -DSOME_MACRO=1 -fsyntax-only -Werror

// RUN: %clang -cc1depscan -o %t/pch.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -x c-header %t/prefix.h -emit-pch -DSOME_MACRO=1 -fcas-path %t/cas -Werror
// RUN: %clang @%t/pch.rsp -o %t/prefix2.pch

// RUN: %clang -cc1depscan -o %t/tu.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 %t/t1.c -fsyntax-only -include-pch %t/prefix2.pch -DSOME_MACRO=1 -fcas-path %t/cas -Werror
// RUN: %clang @%t/tu.rsp

//--- t1.c

//--- prefix.h
#undef SOME_MACRO
#define SOME_MACRO 0

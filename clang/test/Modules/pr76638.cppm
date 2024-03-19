// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod1.cppm -emit-module-interface -o %t/mod1.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod2.cppm -fmodule-file=mod1=%t/mod1.pcm \
// RUN:     -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/mod3.cppm -emit-module-interface -o %t/mod3.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod4.cppm -fmodule-file=mod3=%t/mod3.pcm \
// RUN:     -fsyntax-only -verify

// Testing the behavior of `-fskip-odr-check-in-gmf`
// RUN: %clang_cc1 -std=c++20 %t/mod3.cppm -fskip-odr-check-in-gmf \
// RUN:     -emit-module-interface -o %t/mod3.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod4.cppm -fmodule-file=mod3=%t/mod3.pcm \
// RUN:     -fskip-odr-check-in-gmf -DSKIP_ODR_CHECK_IN_GMF -fsyntax-only -verify

//--- size_t.h

extern "C" {
    typedef unsigned int size_t;
}

//--- csize_t
namespace std {
            using :: size_t;
}

//--- align.h
namespace std {
    enum class align_val_t : size_t {};
}

//--- mod1.cppm
module;
#include "size_t.h"
#include "align.h"
export module mod1;
export using std::align_val_t;

//--- mod2.cppm
// expected-no-diagnostics
module;
#include "size_t.h"
#include "csize_t"
#include "align.h"
export module mod2;
import mod1;
export using std::align_val_t;

//--- signed_size_t.h
// Test that we can still find the case if the underlying type is different
extern "C" {
    typedef signed int size_t;
}

//--- mod3.cppm
module;
#include "size_t.h"
#include "align.h"
export module mod3;
export using std::align_val_t;

//--- mod4.cppm
module;
#include "signed_size_t.h"
#include "csize_t"
#include "align.h"
export module mod4;
import mod3;
export using std::align_val_t;

#ifdef SKIP_ODR_CHECK_IN_GMF
// expected-no-diagnostics
#else
// expected-error@align.h:* {{'std::align_val_t' has different definitions in different modules; defined here first difference is enum with specified type 'size_t' (aka 'int')}}
// expected-note@align.h:* {{but in 'mod3.<global>' found enum with specified type 'size_t' (aka 'unsigned int')}}
#endif

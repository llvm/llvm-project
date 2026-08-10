// RUN: %clang_cc1 -E -std=c++11 -o - %s | FileCheck %s

#define id(x) x
id("s")_x // CHECK: "s" _x
id(L"s")_x // CHECK: L"s" _x
id(u8"s")_x // CHECK: u8"s" _x
id(u"s")_x // CHECK: u"s" _x
id(U"s")_x // CHECK: U"s" _x
id('s')_x // CHECK: 's' _x
id(L's')_x // CHECK: L's' _x
id(u's')_x // CHECK: u's' _x
id(U's')_x // CHECK: U's' _x
id("s"_x)_y // CHECK: "s"_x _y
id(1.0_)f // CHECK: 1.0_ f
id(1.0)_f // CHECK: 1.0 _f
id(0xface+)b_count // CHECK: 0xface+ b_count
id("s")1 // CHECK: "s"1
id("s"_x)1 // CHECK: "s"_x 1
id(1)_2_x // CHECK: 1 _2_x

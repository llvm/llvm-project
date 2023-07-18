// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -triple i686-pc-win32 -Werror -Wno-address-of-temporary -D"Class=struct objc_class *" -D"id=struct objc_object *" -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp -Wno-attributes
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-win32 -Werror -Wno-address-of-temporary -D"Class=struct objc_class *" -D"id=struct objc_object *" -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp -Wno-attributes
// rdar://14913632

extern "C" void *sel_registerName(const char *);

void x() {
    id y;
    for (id a in y) {
    }
}

// CHECK: #ifdef _WIN64
// CHECK-NEXT: typedef unsigned long long  _WIN_NSUInteger;
// CHECK-NEXT: #else
// CHECK-NEXT: typedef unsigned int _WIN_NSUInteger;
// CHECK-NEXT: #endif
// CHECK: _WIN_NSUInteger limit =
// CHECK-NEXT: ((_WIN_NSUInteger (*) (id, SEL, struct __objcFastEnumerationState *, id *, _WIN_NSUInteger))(void *)objc_msgSend)
// CHECK-NEXT: ((id)l_collection,
// CHECK-NEXT: sel_registerName("countByEnumeratingWithState:objects:count:"),
// CHECK-NEXT: &enumState, (id *)__rw_items, (_WIN_NSUInteger)16);

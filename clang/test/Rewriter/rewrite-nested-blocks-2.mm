// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// grep "static void __FUNC_block_copy_" %t-rw.cpp | count 2
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// grep "static void __FUNC_block_copy_" %t-modern-rw.cpp | count 2

typedef unsigned long size_t;
void Outer(void (^bk)());
void Inner(void (^bk)());
void INNER_FUNC(id d);

void FUNC() {
    
    id bar = (id)42;
    Outer(^{
        Inner(^{
            INNER_FUNC(bar);
        });
    });    
}

// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s
// rdar://9006279

void q(void (^p)(void)) {
    p();
}

void f() {
    __block char BYREF_VAR_CHECK = 'a';
    __block char d = 'd';
    q(^{
        q(^{
            __block char e = 'e';
            char l = 'l';
            BYREF_VAR_CHECK = 'b';
            d = 'd';
            q(^{
                 e = '1';
                 BYREF_VAR_CHECK = '2';
                 d = '3';
               }
             );
        });
    });
}

int main() {
    f();
    return 0;
}

// CHECK: (__Block_byref_BYREF_VAR_CHECK_0 *)BYREF_VAR_CHECK
// CHECK: {(void*)0,(__Block_byref_BYREF_VAR_CHECK_0 *)&BYREF_VAR_CHECK, 0, sizeof(__Block_byref_BYREF_VAR_CHECK_0), 'a'}
// CHECK: __Block_byref_BYREF_VAR_CHECK_0 *)&BYREF_VAR_CHECK, (__Block_byref_d_1 *)&d, 570425344)));

// RUN: %clang_cc1 -x c -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck --input-file=%t-rw.cpp %s

void q(void (^p)(void)) {
    p();
}

void f(void) {
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

int main(void) {
    f();
    return 0;
}

// CHECK: (__Block_byref_BYREF_VAR_CHECK_0 *)BYREF_VAR_CHECK
// CHECK: (__Block_byref_BYREF_VAR_CHECK_0 *)&BYREF_VAR_CHECK
// CHECK: (struct __Block_byref_BYREF_VAR_CHECK_0 *)&BYREF_VAR_CHECK, (struct __Block_byref_d_1 *)&d, 570425344));

// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=gnu23 -fdefer-ts -fsyntax-only -fblocks -verify %s

typedef void** jmp_buf;
typedef void** sigjmp_buf;

int setjmp(jmp_buf env);
int _setjmp(jmp_buf env);
int sigsetjmp(sigjmp_buf env, int savesigs);
int __sigsetjmp(sigjmp_buf env, int savesigs);
void longjmp(jmp_buf env, int val);
void _longjmp(jmp_buf env, int val);
void siglongjmp(sigjmp_buf env, int val);

jmp_buf x;
sigjmp_buf y;
void f() {
    _Defer {
        __builtin_setjmp(x); // expected-error {{cannot use '__builtin_setjmp' inside a defer statement}}
        __builtin_longjmp(x, 1); // expected-error {{cannot use '__builtin_longjmp' inside a defer statement}}
        setjmp(x); // expected-error {{cannot use 'setjmp' inside a defer statement}}
        _setjmp(x); // expected-error {{cannot use '_setjmp' inside a defer statement}}
        sigsetjmp(y, 0); // expected-error {{cannot use 'sigsetjmp' inside a defer statement}}
        __sigsetjmp(y, 0); // expected-error {{cannot use '__sigsetjmp' inside a defer statement}}
        longjmp(x, 0); // expected-error {{cannot use 'longjmp' inside a defer statement}}
        _longjmp(x, 0); // expected-error {{cannot use '_longjmp' inside a defer statement}}
        siglongjmp(y, 0); // expected-error {{cannot use 'siglongjmp' inside a defer statement}}

        (void) ^{
            __builtin_setjmp(x);
            __builtin_longjmp(x, 1);
            setjmp(x);
            _setjmp(x);
            sigsetjmp(y, 0);
            __sigsetjmp(y, 0);
            longjmp(x, 0);
            _longjmp(x, 0);
            siglongjmp(y, 0);

            _Defer {
                __builtin_setjmp(x); // expected-error {{cannot use '__builtin_setjmp' inside a defer statement}}
                __builtin_longjmp(x, 1); // expected-error {{cannot use '__builtin_longjmp' inside a defer statement}}
                setjmp(x); // expected-error {{cannot use 'setjmp' inside a defer statement}}
                _setjmp(x); // expected-error {{cannot use '_setjmp' inside a defer statement}}
                sigsetjmp(y, 0); // expected-error {{cannot use 'sigsetjmp' inside a defer statement}}
                __sigsetjmp(y, 0); // expected-error {{cannot use '__sigsetjmp' inside a defer statement}}
                longjmp(x, 0); // expected-error {{cannot use 'longjmp' inside a defer statement}}
                _longjmp(x, 0); // expected-error {{cannot use '_longjmp' inside a defer statement}}
                siglongjmp(y, 0); // expected-error {{cannot use 'siglongjmp' inside a defer statement}}
            }
        };
    }
}

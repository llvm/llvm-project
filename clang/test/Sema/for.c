// RUN: %clang_cc1 -fsyntax-only -verify=c11 -std=c11 -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify=c23 -std=c23 -Wpre-c23-compat %s

// Check C99 6.8.5p3
void b1 (void) { for (void (*f) (void);;); }
void b2 (void) { for (void f (void);;); }   /* c11-warning {{non-variable declaration in 'for' loop is a C23 extension}}
                                               c23-warning {{non-variable declaration in 'for' loop is incompatible with C standards before C23}} */
void b3 (void) { for (static int f;f;); }    /* c11-warning {{declaration of non-local variable in 'for' loop is a C23 extension}}
                                                c23-warning {{declaration of non-local variable in 'for' loop is incompatible with C standards before C23}} */

void b4 (void) { for (typedef int f;;); }   /* c11-warning {{non-variable declaration in 'for' loop is a C23 extension}}
                                               c23-warning {{non-variable declaration in 'for' loop is incompatible with C standards before C23}} */
void b5 (void) { for (struct { int i; } s;s.i;); }
void b6 (void) { for (enum { zero, ten = 10 } i = ten;i;); }
void b7 (void) { for (struct s { int i; };;); } /* c11-warning {{non-variable declaration in 'for' loop is a C23 extension}}
                                                   c23-warning {{non-variable declaration in 'for' loop is incompatible with C standards before C23}} */
void b8 (void) { for (static struct { int i; } s;s.i;); } /* c11-warning {{declaration of non-local variable in 'for' loop is a C23 extension}}
                                                             c23-warning {{declaration of non-local variable in 'for' loop is incompatible with C standards before C23}} */
void b9 (void) { for (struct { int i; } (*s)(struct { int j; } o) = 0;s;); }
void b10(void) { for (typedef struct { int i; } (*s)(struct { int j; });;); } /* c11-warning {{non-variable declaration in 'for' loop is a C23 extension}}
                                                                                 c23-warning {{non-variable declaration in 'for' loop is incompatible with C standards before C23}} */

#if __has_feature(c_thread_local)
void b11 (void) { for (static _Thread_local struct { int i; } s;s.i;); } /* c11-warning {{declaration of non-local variable in 'for' loop is a C23 extension}}
                                                                            c23-warning {{declaration of non-local variable in 'for' loop is incompatible with C standards before C23}} */
#endif

void b12(void) {
  for(_Static_assert(1, "");;) {} /* c11-warning {{non-variable declaration in 'for' loop is a C23 extension}}
                                     c23-warning {{non-variable declaration in 'for' loop is incompatible with C standards before C23}} */
}

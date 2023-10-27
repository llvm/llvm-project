// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsyntax-only -verify %s
// PR6913

int main(void)
{
   int x[10][10];
   int (*p)[] = x;

   int i;

   for(i = 0; i < 10; ++i)
   {
       p[i][i] = i; // expected-error {{subscript of pointer to incomplete type 'int[]'}}
   }
}

void foo(int a[*][2]) {(void)a[0][1]; } // expected-error {{variable length array must be bound in function definition}}
void foo1(int a[2][*]) {(void)a[0][1]; } // expected-error {{variable length array must be bound in function definition}}
void foo2(int a[*][*]) {(void)a[0][1]; } // expected-error {{variable length array must be bound in function definition}}
void foo3(int a[2][*][2]) {(void)a[0][1][1]; } // expected-error {{variable length array must be bound in function definition}}
void foo4(int a[2][*][*]) {(void)a[0][1][1]; } // expected-error {{variable length array must be bound in function definition}}

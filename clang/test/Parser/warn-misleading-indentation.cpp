// RUN: %clang_cc1 -x c -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -Wall -Wno-unused -Wno-misleading-indentation -DCXX17 %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wmisleading-indentation -DWITH_WARN -ftabstop 8 -DTAB_SIZE=8 %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -Wall -Wno-unused -DWITH_WARN  -ftabstop 4 -DTAB_SIZE=4 -DCXX17 %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wall -Wno-unused -DWITH_WARN -ftabstop 1 -DTAB_SIZE=1 %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -Wall -Wno-unused -Wmisleading-indentation -DCXX17 -DWITH_WARN -ftabstop 2 -DTAB_SIZE=2 %s

#ifndef WITH_WARN
// expected-no-diagnostics
#endif

void f0(int i) {
  if (i)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
    i = i + 1;
    int x = 0;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
#endif
  return;
#ifdef CXX17
  if constexpr (false)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
    i = 0;
    i += 1;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
#endif
#endif
}

void f1(int i) {
  for (;i;)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
    i = i + 1;
    i *= 2;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'for'}}
#endif
  return;
}

void f2(int i) {
  while (i)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
    i = i + 1; i *= 2;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'while'}}
#endif
  return;
}

void f3(int i) {
  if (i)
    i = i + 1;
  else
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
    i *= 2;
    const int x = 0;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'else'}}
#endif
}

#ifdef CXX17
struct Range {
  int *begin() {return nullptr;}
  int *end() {return nullptr;}
};
#endif

void f4(int i) {
  if (i)
  i *= 2;
  return;
  if (i)
    i *= 2;
    ;
  if (i)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
    i *= 2;
    typedef int Int;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
#endif
#ifdef CXX17
  Range R;
  for (auto e : R)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
    i *= 2;
    using Int2 = int;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'for'}}
#endif
#endif
}

int bar(void);

int foo(int* dst)
{   
    if (dst)
       return
    bar();
  if (dst)
    dst = dst + \
    bar();
  return 0;
}

void g(int i) {
  if (1)
    i = 2;
  else
         if (i == 3)
#ifdef WITH_WARN
// expected-note@-3 {{here}}
#endif
    i = 4;
    i = 5;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
#endif
}

// Or this
#define TEST i = 5
void g0(int i) {
  if (1)
    i = 2;
  else
    i = 5;
    TEST;
}

void g1(int i) {
  if (1)
    i = 2;
  else if (i == 3)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
      i = 4;
      i = 5;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
#endif
}

void g2(int i) {
  if (1)
    i = 2;
  else
    if (i == 3)
    {i = 4;}
    i = 5;
}

void g6(int i) {
        if (1)
                if (i == 3)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
                        i = 4;
                        i = 5;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
#endif
}

void g7(int i) {
  if (1)
    i = 4;
#ifdef TEST1
#endif
    i = 5;
}

void a1(int i) { if (1) i = 4; return; }

void a2(int i) {
  {
    if (1)
      i = 4;
      }
  return;
}

void a3(int i) {
  if (1)
    {
    i = 4;
    }
    return;
}

void s(int num) {
    {
        if (1)
            return;
        else
            return;
        return;
    }
    if (0)
#ifdef WITH_WARN
// expected-note@-2 {{here}}
#endif
        return;
        return;
#ifdef WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
#endif
}
int a4()
{
	if (0)
		return 1;
 	return 0;
#if (TAB_SIZE == 1)
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
// expected-note@-5 {{here}}
#endif 
}

int a5()
{
	if (0)
		return 1;
		return 0;
#if WITH_WARN
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
// expected-note@-5 {{here}}
#endif
}

int a6()
{
	if (0)
		return 1;
      		return 0;
#if (TAB_SIZE == 8)
// expected-warning@-2 {{misleading indentation; statement is not part of the previous 'if'}}
// expected-note@-5 {{here}}
#endif
}

#define FOO \
 goto fail

int main(int argc, char* argv[]) {
  if (5 != 0)
    goto fail;
  else
    goto fail;

  if (1) {
    if (1)
      goto fail;
    else if (1)
      goto fail;
    else if (1)
      goto fail;
    else
      goto fail;
  } else if (1) {
    if (1)
      goto fail;
  }

  if (1) {
    if (1)
      goto fail;
  } else if (1)
    goto fail;


  if (1) goto fail; goto fail;

    if (0)
        goto fail;

    goto fail;

    if (0)
        FOO;

    goto fail;

fail:;
}

void f_label(int b) {
  if (b)
    return;
    a:
  return;
  goto a;
}

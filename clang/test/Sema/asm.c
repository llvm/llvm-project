// RUN: %clang_cc1 %s -Wno-private-extern -triple i386-pc-linux-gnu -verify -fsyntax-only

void f() {
  int i;

  asm ("foo\n" : : "a" (i + 2));
  asm ("foo\n" : : "a" (f())); // expected-error {{invalid type 'void' in asm input}}

  asm ("foo\n" : "=a" (f())); // expected-error {{invalid lvalue in asm output}}
  asm ("foo\n" : "=a" (i + 2)); // expected-error {{invalid lvalue in asm output}}

  asm ("foo\n" : [symbolic_name] "=a" (i) : "[symbolic_name]" (i));
  asm ("foo\n" : "=a" (i) : "[" (i)); // expected-error {{invalid input constraint '[' in asm}}
  asm ("foo\n" : "=a" (i) : "[foo" (i)); // expected-error {{invalid input constraint '[foo' in asm}}
  asm ("foo\n" : "=a" (i) : "[symbolic_name]" (i)); // expected-error {{invalid input constraint '[symbolic_name]' in asm}}
}

void clobbers() {
  asm ("nop" : : : "ax", "#ax", "%ax");
  asm ("nop" : : : "eax", "rax", "ah", "al");
  asm ("nop" : : : "0", "%0", "#0");
  asm ("nop" : : : "foo"); // expected-error {{unknown register name 'foo' in asm}}
  asm ("nop" : : : "52");
  asm ("nop" : : : "104"); // expected-error {{unknown register name '104' in asm}}
  asm ("nop" : : : "-1"); // expected-error {{unknown register name '-1' in asm}}
  asm ("nop" : : : "+1"); // expected-error {{unknown register name '+1' in asm}}
}

// rdar://6094010
void test3() {
  int x;
  asm(L"foo" : "=r"(x)); // expected-error {{wide string}}
  asm("foo" : L"=r"(x)); // expected-error {{wide string}}
}

// <rdar://problem/6156893>
void test4(const volatile void *addr)
{
    asm ("nop" : : "r"(*addr)); // expected-error {{invalid type 'const volatile void' in asm input for constraint 'r'}}
    asm ("nop" : : "m"(*addr));

    asm ("nop" : : "r"(test4(addr))); // expected-error {{invalid type 'void' in asm input for constraint 'r'}}
    asm ("nop" : : "m"(test4(addr))); // expected-error {{invalid lvalue in asm input for constraint 'm'}}

    asm ("nop" : : "m"(f())); // expected-error {{invalid lvalue in asm input for constraint 'm'}}
}

// <rdar://problem/6512595>
void test5() {
  asm("nop" : : "X" (8));
}

// PR3385
void test6(long i) {
  asm("nop" : : "er"(i));
}

void asm_string_tests(int i) {
  asm("%!");   // simple asm string, %! is not an error.
  asm("%!" : );   // expected-error {{invalid % escape in inline assembly string}}
  asm("xyz %" : );   // expected-error {{invalid % escape in inline assembly string}}

  asm ("%[somename]" :: [somename] "i"(4)); // ok
  asm ("%[somename]" :: "i"(4)); // expected-error {{unknown symbolic operand name in inline assembly string}}
  asm ("%[somename" :: "i"(4)); // expected-error {{unterminated symbolic operand name in inline assembly string}}
  asm ("%[]" :: "i"(4)); // expected-error {{empty symbolic operand name in inline assembly string}}

  // PR3258
  asm("%9" :: "i"(4)); // expected-error {{invalid operand number in inline asm string}}
  asm("%1" : "+r"(i)); // ok, referring to input.
}

// PR4077
int test7(unsigned long long b) {
  int a;
  asm volatile("foo %0 %1" : "=a" (a) :"0" (b)); // expected-error {{input with type 'unsigned long long' matching output with type 'int'}}
  return a;
}

// <rdar://problem/7574870>
asm volatile (""); // expected-warning {{meaningless 'volatile' on asm outside function}}

// PR3904
void test8(int i) {
  // A number in an input constraint can't point to a read-write constraint.
  asm("" : "+r" (i), "=r"(i) :  "0" (i)); // expected-error{{invalid input constraint '0' in asm}}
}

// PR3905
void test9(int i) {
  asm("" : [foo] "=r" (i), "=r"(i) : "1[foo]"(i)); // expected-error{{invalid input constraint '1[foo]' in asm}}
  asm("" : [foo] "=r" (i), "=r"(i) : "[foo]1"(i)); // expected-error{{invalid input constraint '[foo]1' in asm}}
}

register int g asm("dx"); // expected-error{{global register variables are not supported}}

void test10(void){
  static int g asm ("g_asm") = 0;
  extern int gg asm ("gg_asm");
  __private_extern__ int ggg asm ("ggg_asm");

  int a asm ("a_asm"); // expected-warning{{ignored asm label 'a_asm' on automatic variable}}
  auto int aa asm ("aa_asm"); // expected-warning{{ignored asm label 'aa_asm' on automatic variable}}

  register int r asm ("cx");
  register int rr asm ("rr_asm"); // expected-error{{unknown register name 'rr_asm' in asm}}
}

// This is just an assert because of the boolean conversion.
// Feel free to change the assembly to something sensible if it causes a problem.
// rdar://problem/9414925
void test11(void) {
  _Bool b;
  asm volatile ("movb %%gs:%P2,%b0" : "=q"(b) : "0"(0), "i"(5L));
}

void test12(void) {
  register int cc __asm ("cc"); // expected-error{{unknown register name 'cc' in asm}}
}

// PR10223
void test13(void) {
  void *esp;
  __asm__ volatile ("mov %%esp, %o" : "=r"(esp) : : ); // expected-error {{invalid % escape in inline assembly string}}
}

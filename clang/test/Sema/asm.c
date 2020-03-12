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

  asm ("foo\n" : : "" (i)); // expected-error {{invalid input constraint '' in asm}}
  asm ("foo\n" : "=a" (i) : "" (i)); // expected-error {{invalid input constraint '' in asm}}
}

void clobbers() {
  asm ("nop" : : : "ax", "#ax", "%ax");
  asm ("nop" : : : "eax", "rax", "ah", "al");
  asm ("nop" : : : "0", "%0", "#0");
  asm ("nop" : : : "foo"); // expected-error {{unknown register name 'foo' in asm}}
  asm ("nop" : : : "52");
  asm ("nop" : : : "204"); // expected-error {{unknown register name '204' in asm}}
  asm ("nop" : : : "-1"); // expected-error {{unknown register name '-1' in asm}}
  asm ("nop" : : : "+1"); // expected-error {{unknown register name '+1' in asm}}
  register void *clobber_conflict asm ("%rcx");
  register void *no_clobber_conflict asm ("%rax");
  int a,b,c;
  asm ("nop" : "=r" (no_clobber_conflict) : "r" (clobber_conflict) : "%rcx"); // expected-error {{asm-specifier for input or output variable conflicts with asm clobber list}}
  asm ("nop" : "=r" (clobber_conflict) : "r" (no_clobber_conflict) : "%rcx"); // expected-error {{asm-specifier for input or output variable conflicts with asm clobber list}}
  asm ("nop" : "=r" (clobber_conflict) : "r" (clobber_conflict) : "%rcx"); // expected-error {{asm-specifier for input or output variable conflicts with asm clobber list}}
  asm ("nop" : "=c" (a) : "r" (no_clobber_conflict) : "%rcx"); // expected-error {{asm-specifier for input or output variable conflicts with asm clobber list}}
  asm ("nop" : "=r" (no_clobber_conflict) : "c" (c) : "%rcx"); // expected-error {{asm-specifier for input or output variable conflicts with asm clobber list}}
  asm ("nop" : "=r" (clobber_conflict) : "c" (c) : "%rcx"); // expected-error {{asm-specifier for input or output variable conflicts with asm clobber list}}
  asm ("nop" : "=a" (a) : "b" (b) : "%rcx", "%rbx"); // expected-error {{asm-specifier for input or output variable conflicts with asm clobber list}} 
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

void test10(void){
  static int g asm ("g_asm") = 0;
  extern int gg asm ("gg_asm");
  __private_extern__ int ggg asm ("ggg_asm");

  int a asm ("a_asm"); // expected-warning{{ignored asm label 'a_asm' on automatic variable}}
  auto int aa asm ("aa_asm"); // expected-warning{{ignored asm label 'aa_asm' on automatic variable}}

  register int r asm ("cx");
  register int rr asm ("rr_asm"); // expected-error{{unknown register name 'rr_asm' in asm}}
  register int rrr asm ("%"); // expected-error{{unknown register name '%' in asm}}
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

// <rdar://problem/12700799>
struct S;  // expected-note 2 {{forward declaration of 'struct S'}}
void test14(struct S *s) {
  __asm("": : "a"(*s)); // expected-error {{dereference of pointer to incomplete type 'struct S'}}
  __asm("": "=a" (*s) :); // expected-error {{dereference of pointer to incomplete type 'struct S'}}
}

// PR15759.
double test15() {
  double ret = 0;
  __asm("0.0":"="(ret)); // expected-error {{invalid output constraint '=' in asm}}
  __asm("0.0":"=&"(ret)); // expected-error {{invalid output constraint '=&' in asm}}
  __asm("0.0":"+?"(ret)); // expected-error {{invalid output constraint '+?' in asm}}
  __asm("0.0":"+!"(ret)); // expected-error {{invalid output constraint '+!' in asm}}
  __asm("0.0":"+#"(ret)); // expected-error {{invalid output constraint '+#' in asm}}
  __asm("0.0":"+*"(ret)); // expected-error {{invalid output constraint '+*' in asm}}
  __asm("0.0":"=%"(ret)); // expected-error {{invalid output constraint '=%' in asm}}
  __asm("0.0":"=,="(ret)); // expected-error {{invalid output constraint '=,=' in asm}}
  __asm("0.0":"=,g"(ret)); // no-error
  __asm("0.0":"=g"(ret)); // no-error
  return ret;
}

void iOutputConstraint(int x){
  __asm ("nop" : "=ir" (x) : :); // no-error
  __asm ("nop" : "=ri" (x) : :); // no-error
  __asm ("nop" : "=ig" (x) : :); // no-error
  __asm ("nop" : "=im" (x) : :); // no-error
  __asm ("nop" : "=imr" (x) : :); // no-error
  __asm ("nop" : "=i" (x) : :); // expected-error{{invalid output constraint '=i' in asm}}
  __asm ("nop" : "+i" (x) : :); // expected-error{{invalid output constraint '+i' in asm}}
  __asm ("nop" : "=ii" (x) : :); // expected-error{{invalid output constraint '=ii' in asm}}
  __asm ("nop" : "=nr" (x) : :); // no-error
  __asm ("nop" : "=rn" (x) : :); // no-error
  __asm ("nop" : "=ng" (x) : :); // no-error
  __asm ("nop" : "=nm" (x) : :); // no-error
  __asm ("nop" : "=nmr" (x) : :); // no-error
  __asm ("nop" : "=n" (x) : :); // expected-error{{invalid output constraint '=n' in asm}}
  __asm ("nop" : "+n" (x) : :); // expected-error{{invalid output constraint '+n' in asm}}
  __asm ("nop" : "=nn" (x) : :); // expected-error{{invalid output constraint '=nn' in asm}}
  __asm ("nop" : "=Fr" (x) : :); // no-error
  __asm ("nop" : "=rF" (x) : :); // no-error
  __asm ("nop" : "=Fg" (x) : :); // no-error
  __asm ("nop" : "=Fm" (x) : :); // no-error
  __asm ("nop" : "=Fmr" (x) : :); // no-error
  __asm ("nop" : "=F" (x) : :); // expected-error{{invalid output constraint '=F' in asm}}
  __asm ("nop" : "+F" (x) : :); // expected-error{{invalid output constraint '+F' in asm}}
  __asm ("nop" : "=FF" (x) : :); // expected-error{{invalid output constraint '=FF' in asm}}
  __asm ("nop" : "=Er" (x) : :); // no-error
  __asm ("nop" : "=rE" (x) : :); // no-error
  __asm ("nop" : "=Eg" (x) : :); // no-error
  __asm ("nop" : "=Em" (x) : :); // no-error
  __asm ("nop" : "=Emr" (x) : :); // no-error
  __asm ("nop" : "=E" (x) : :); // expected-error{{invalid output constraint '=E' in asm}}
  __asm ("nop" : "+E" (x) : :); // expected-error{{invalid output constraint '+E' in asm}}
  __asm ("nop" : "=EE" (x) : :); // expected-error{{invalid output constraint '=EE' in asm}}
}

// PR19837
struct foo {
  int a;
};
register struct foo bar asm("esp"); // expected-error {{bad type for named register variable}}
register float baz asm("esp"); // expected-error {{bad type for named register variable}}

register int r0 asm ("edi"); // expected-error {{register 'edi' unsuitable for global register variables on this target}}
register long long r1 asm ("esp"); // expected-error {{size of register 'esp' does not match variable size}}
register int r2 asm ("esp");

double f_output_constraint(void) {
  double result;
  __asm("foo1": "=f" (result)); // expected-error {{invalid output constraint '=f' in asm}}
  return result;
}

void fn1() {
  int l;
  __asm__(""
          : [l] "=r"(l)
          : "[l],m"(l)); // expected-error {{asm constraint has an unexpected number of alternatives: 1 vs 2}}
}

void fn2() {
  int l;
 __asm__(""
          : "+&m"(l)); // expected-error {{invalid output constraint '+&m' in asm}}
}

void fn3() {
  int l;
 __asm__(""
          : "+#r"(l)); // expected-error {{invalid output constraint '+#r' in asm}}
}

void fn4() {
  int l;
 __asm__(""
          : "=r"(l)
          : "m#"(l));
}

void fn5() {
  int l;
    __asm__(""
          : [g] "+r"(l)
          : "[g]"(l)); // expected-error {{invalid input constraint '[g]' in asm}}
}

void fn6() {
    int a;
  __asm__(""
            : "=rm"(a), "=rm"(a)
            : "11m"(a)); // expected-error {{invalid input constraint '11m' in asm}}
}

// PR14269
typedef struct test16_foo {
  unsigned int field1 : 1;
  unsigned int field2 : 2;
  unsigned int field3 : 3;
} test16_foo;
typedef __attribute__((vector_size(16))) int test16_bar;
register int test16_baz asm("esp");

void test16()
{
  test16_foo a;
  test16_bar b;

  __asm__("movl $5, %0"
          : "=rm" (a.field2)); // expected-error {{reference to a bit-field in asm input with a memory constraint '=rm'}}
  __asm__("movl $5, %0"
          :
          : "m" (a.field3)); // expected-error {{reference to a bit-field in asm output with a memory constraint 'm'}}
  __asm__("movl $5, %0"
          : "=rm" (b[2])); // expected-error {{reference to a vector element in asm input with a memory constraint '=rm'}}
  __asm__("movl $5, %0"
          :
          : "m" (b[3])); // expected-error {{reference to a vector element in asm output with a memory constraint 'm'}}
  __asm__("movl $5, %0"
          : "=rm" (test16_baz)); // expected-error {{reference to a global register variable in asm input with a memory constraint '=rm'}}
  __asm__("movl $5, %0"
          :
          : "m" (test16_baz)); // expected-error {{reference to a global register variable in asm output with a memory constraint 'm'}}
}

int test17(int t0)
{
  int r0, r1;
  __asm ("addl %2, %2\n\t"
         "movl $123, %0"
         : "=a" (r0),
           "=&r" (r1)
         : "1" (t0),   // expected-note {{constraint '1' is already present here}}
           "1" (t0));  // expected-error {{more than one input constraint matches the same output '1'}}
  return r0 + r1;
}

void test18()
{
  // expected-error@+2 {{duplicate use of asm operand name "lab"}}
  // expected-note@+1 {{asm operand name "lab" first referenced here}}
  asm goto ("" : : : : lab, lab, lab2, lab);
  // expected-error@+2 {{duplicate use of asm operand name "lab"}}
  // expected-note@+1 {{asm operand name "lab" first referenced here}}
  asm goto ("xorw %[lab], %[lab]; je %l[lab]" : : [lab] "i" (0) : : lab);
lab:;
lab2:;
  int x,x1;
  // expected-error@+2 {{duplicate use of asm operand name "lab"}}
  // expected-note@+1 {{asm operand name "lab" first referenced here}}
  asm ("" : [lab] "=r" (x),[lab] "+r" (x) : [lab1] "r" (x));
  // expected-error@+2 {{duplicate use of asm operand name "lab"}}
  // expected-note@+1 {{asm operand name "lab" first referenced here}}
  asm ("" : [lab] "=r" (x1) : [lab] "r" (x));
  // expected-error@+1 {{invalid operand number in inline asm string}}
  asm ("jne %l0":::);
  asm goto ("jne %l0"::::lab);
}

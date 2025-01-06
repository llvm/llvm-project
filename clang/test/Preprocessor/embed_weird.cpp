// RUN: rm -rf %t && mkdir -p %t/media
// RUN: cp %S/Inputs/single_byte.txt %S/Inputs/jk.txt %S/Inputs/numbers.txt %t/
// RUN: cp %S/Inputs/media/empty %t/media/
// RUN: printf "\0" > %t/null_byte.bin
// RUN: %clang_cc1 %s -fsyntax-only --embed-dir=%t -verify=expected,cxx -Wno-c23-extensions
// RUN: %clang_cc1 -x c -std=c23 %s -fsyntax-only --embed-dir=%t -verify=expected,c
// RUN: %clang_cc1 %s -fsyntax-only -fexperimental-new-constant-interpreter --embed-dir=%t -verify=expected,cxx -Wno-c23-extensions
// RUN: %clang_cc1 -x c -std=c23 %s -fsyntax-only -fexperimental-new-constant-interpreter --embed-dir=%t -verify=expected,c
#embed <media/empty>
;

void f (unsigned char x) { (void)x;}
void g () {}
void h (unsigned char x, int y) {(void)x; (void)y;}
int i () {
	return
#embed <single_byte.txt>
		;
}

_Static_assert(
#embed <single_byte.txt> suffix(,)
""
);
_Static_assert(
#embed <single_byte.txt>
, ""
);
_Static_assert(sizeof(
#embed <single_byte.txt>
) ==
sizeof(int)
, ""
);
_Static_assert(sizeof
#embed <single_byte.txt>
, ""
);
_Static_assert(sizeof(
#embed <jk.txt>
) ==
sizeof(int)
, ""
);

#ifdef __cplusplus
template <int First, int Second>
void j() {
	static_assert(First == 'j', "");
	static_assert(Second == 'k', "");
}
#endif

void do_stuff() {
	f(
#embed <single_byte.txt>
	);
	g(
#embed <media/empty>
	);
	h(
#embed <jk.txt>
	);
	int r = i();
	(void)r;
#ifdef __cplusplus
	j<
#embed <jk.txt>
	>(
#embed <media/empty>
	);
#endif
}

// Ensure that we don't accidentally allow you to initialize an unsigned char *
// from embedded data; the data is modeled as a string literal internally, but
// is not actually a string literal.
const unsigned char *ptr = (
#embed <jk.txt> // expected-warning {{left operand of comma operator has no effect}}
    ); // c-error@-2 {{incompatible integer to pointer conversion initializing 'const unsigned char *' with an expression of type 'int'}} \
     cxx-error@-2 {{cannot initialize a variable of type 'const unsigned char *' with an rvalue of type 'int'}}

// However, there are some cases where this is fine and should work.
const unsigned char *null_ptr_1 =
#embed <media/empty> if_empty(0)
;

const unsigned char *null_ptr_2 =
#embed <null_byte.bin>
;

const unsigned char *null_ptr_3 = {
#embed <null_byte.bin>
};

#define FILE_NAME <null_byte.bin>
#define LIMIT 1
#define OFFSET 0
#define EMPTY_SUFFIX suffix()

constexpr unsigned char ch =
#embed FILE_NAME limit(LIMIT) clang::offset(OFFSET) EMPTY_SUFFIX
;
static_assert(ch == 0);

void foobar(float x, char y, char z);
void g1() { foobar((float)
#embed "numbers.txt" limit(3)
);
}

#if __cplusplus
struct S { S(char x); ~S(); };
void f1() {
  S s[] = {
#embed "null_byte.bin"
  };
}
#endif

static_assert(_Generic(
#embed __FILE__ limit(1)
  , int : 1, default : 0));

static_assert(alignof(typeof(
#embed __FILE__ limit(1)
)) == alignof(int));

struct HasChar {
  signed char ch;
};

constexpr struct HasChar c = {
#embed "Inputs/big_char.txt" // cxx-error {{constant expression evaluates to 255 which cannot be narrowed to type 'signed char'}} \
                                cxx-note {{insert an explicit cast to silence this issue}} \
                                c-error {{constexpr initializer evaluates to 255 which is not exactly representable in type 'signed char'}}

};

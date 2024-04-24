// RUN: %clang_cc1 %s -fsyntax-only --embed-dir=%S/Inputs -verify=expected,cxx -Wno-c23-extensions
// RUN: %clang_cc1 -x c -std=c23 %s -fsyntax-only --embed-dir=%S/Inputs -verify=expected,c
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
sizeof(unsigned char)
, ""
);
_Static_assert(sizeof
#embed <single_byte.txt>
, ""
);
_Static_assert(sizeof(
#embed <jk.txt>
) ==
sizeof(unsigned char)
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
const unsigned char *ptr =
#embed <jk.txt>
; // c-error@-2 {{incompatible integer to pointer conversion initializing 'const unsigned char *' with an expression of type 'unsigned char'}} \
     cxx-error@-2 {{cannot initialize a variable of type 'const unsigned char *' with an rvalue of type 'unsigned char'}}

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

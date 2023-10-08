// RUN: %clang_cc1 %s -fsyntax-only -embed-dir=%S/Inputs -verify
// RUN: %clang_cc1 -x c %s -fsyntax-only -embed-dir=%S/Inputs -verify
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
// expected-no-diagnostics

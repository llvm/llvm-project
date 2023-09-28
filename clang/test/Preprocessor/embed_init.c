// RUN: %clang_cc1 %s -fsyntax-only -embed-dir=%S/Inputs -CC -verify

typedef struct kitty {
	int purr;
} kitty;

typedef struct kitty_kitty {
	int here;
	kitty kit;
} kitty_kitty;

const int meow =
#embed <single_byte.txt>
;

const kitty kit = {
#embed <single_byte.txt>
};

const kitty_kitty kit_kit = {
#embed <jk.txt>
};

_Static_assert(meow == 'b', "");
_Static_assert(kit.purr == 'b', "");
_Static_assert(kit_kit.here == 'j', "");
_Static_assert(kit_kit.kit.purr == 'k', "");
// expected-no-diagnostics

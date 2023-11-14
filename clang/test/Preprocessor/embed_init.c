// RUN: %clang_cc1 -std=c23 %s -fsyntax-only -embed-dir=%S/Inputs -verify
// expected-no-diagnostics

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

static_assert(meow == 'b');
static_assert(kit.purr == 'b');
static_assert(kit_kit.here == 'j');
static_assert(kit_kit.kit.purr == 'k');

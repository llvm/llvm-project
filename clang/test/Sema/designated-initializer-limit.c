// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-unknown \
// RUN:   -fmax-init-list-elements=4 %s

int designated_at_limit[] = { [3] = 1 };
_Static_assert(sizeof(designated_at_limit) / sizeof(int) == 4, "");

int range_at_limit[] = { [1 ... 3] = 1 };
_Static_assert(sizeof(range_at_limit) / sizeof(int) == 4, "");

int designated_over_limit[] = {
    [4] = 1, // expected-error {{array is too large (5 elements)}}
};

int range_over_limit[] = {
    [1 ... 4] = 1, // expected-error {{array is too large (5 elements)}}
};

int fixed_over_limit[5] = {
    [4] = 1, // expected-error {{array is too large (5 elements)}}
};

int fixed_out_of_bounds[4] = {
    [4] = 1, // expected-error {{array designator index (4) exceeds array bounds (4)}}
};

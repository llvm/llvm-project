void double_declarator1(int *_Nonnull *); // expected-warning{{pointer is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}

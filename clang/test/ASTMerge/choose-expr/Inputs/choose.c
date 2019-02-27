_Static_assert(__builtin_choose_expr(1, 1, 0), "Incorrect semantics of __builtin_choose_expr");
_Static_assert(__builtin_choose_expr(0, 0, 1), "Incorrect semantics of __builtin_choose_expr");

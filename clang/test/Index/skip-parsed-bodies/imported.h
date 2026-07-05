extern int some_val;

static inline int imp_foo() {
  ++some_val; return undef_impval;
}

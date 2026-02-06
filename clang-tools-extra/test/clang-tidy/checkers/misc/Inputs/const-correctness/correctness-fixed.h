struct S {
  void method() const;
  int value;
};

void func_with_ref_param(S const& s);

void func_mixed_params(int x, S const& readonly, S& mutated);

void multi_decl_func(S const& s);
void multi_decl_func(S const& s);

struct S {
  void method() const;
  int value;
};

void func_with_ref_param(S& s);

void func_mixed_params(int x, S& readonly, S& mutated);

void multi_decl_func(S& s);
void multi_decl_func(S& s);

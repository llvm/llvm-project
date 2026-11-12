int i;
int *p = &i;

void use_constant_string_builtins(void) {
  (void)__builtin___CFStringMakeConstantString("");
  (void)__builtin___NSStringMakeConstantString("");
}

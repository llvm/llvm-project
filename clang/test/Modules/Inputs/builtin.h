int i;
int *p = &i;

#ifdef __OBJC__
void use_constant_string_builtins(void) {
  (void)__builtin___CFStringMakeConstantString("");
  (void)__builtin___NSStringMakeConstantString("");
}
#endif


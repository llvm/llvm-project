// Header for PCH test va_arg.c

typedef __builtin_va_list va_list;
char *f (char * (*g) (char **, int), char **p, ...) {
    char *s;
    va_list v;
    s = g (p, __builtin_va_arg(v, int));
    return s;
}

typedef __builtin_ms_va_list __ms_va_list;
char *__attribute__((ms_abi)) h(char *(*i)(char **, int), char **p, ...) {
  char *s;
  __ms_va_list v;
  s = i(p, __builtin_va_arg(v, int));
  return s;
}

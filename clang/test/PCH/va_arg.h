// Header for PCH test va_arg.c

typedef __builtin_va_list va_list;
char *f (char * (*g) (char **, int), char **p, ...) {
    char *s;
    va_list v;
    s = g (p, __builtin_va_arg(v, int));
}

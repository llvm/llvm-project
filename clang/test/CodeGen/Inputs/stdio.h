struct FILE;
extern int vfprintf(struct FILE *s, const char *format, __builtin_va_list arg);
extern int vprintf(const char *format, __builtin_va_list arg);

extern __inline __attribute__((gnu_inline,always_inline)) int
vprintf(const char *x, __builtin_va_list y)
{
  return vfprintf (0, 0, 0);
}

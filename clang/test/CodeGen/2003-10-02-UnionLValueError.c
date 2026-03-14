// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

int sprintf(char * restrict str, const char * restrict format, ...);
union U{
  int i[8];
  char s[80];
};

void format_message(char *buffer, union U *u) {
  sprintf(buffer, u->s);
}

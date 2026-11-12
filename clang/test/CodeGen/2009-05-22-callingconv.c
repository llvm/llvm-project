// RUN: %clang_cc1 %s -emit-llvm -o - -triple i386-unknown-unknown | grep call | grep x86_stdcallcc
void abort(void) __attribute__((__noreturn__));
typedef void re_string_t;
typedef void re_dfa_t;
typedef int reg_errcode_t;
static reg_errcode_t re_string_construct (re_string_t *pstr, const char *str,
       int len, char * trans,
       int icase, const re_dfa_t *dfa)
     __attribute__ ((regparm (3), stdcall));
static reg_errcode_t
re_string_construct (pstr, str, len, trans, icase, dfa)
     re_string_t *pstr;
     const char *str;
     int len, icase;
     char * trans;
     const re_dfa_t *dfa;
{
        if (dfa != (void*)0x282020c0)
                abort();
return 0;
}
int main(void)
{
  return re_string_construct(0, 0, 0, 0, 0, (void*)0x282020c0);
}

// RUN: %check_clang_tidy %s cert-env33-c %t

typedef struct FILE {} FILE;

extern int system(const char *);
extern FILE *popen(const char *, const char *);
extern FILE *_popen(const char *, const char *);

void f(void) {
  // It is permissible to check for the presence of a command processor.
  system(0);

  system("test");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'system' uses a command processor [cert-env33-c]

  popen("test", "test");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling 'popen' uses a command processor
  _popen("test", "test");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: calling '_popen' uses a command processor
}

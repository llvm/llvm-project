__thread int b[2] __attribute__ ((tls_model ("initial-exec")));

extern int foo (void);

int
bar (void)
{
  return foo () + b[0];
}

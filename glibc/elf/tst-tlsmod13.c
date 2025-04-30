__thread int a[2] __attribute__ ((tls_model ("initial-exec")));

int
foo (void)
{
  return a[0];
}

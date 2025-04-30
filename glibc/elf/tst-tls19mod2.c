static int __thread tbar __attribute__ ((tls_model ("initial-exec"))) = 666;

void
setter (int a)
{
  tbar = a;
}

int
bar (void)
{
  return tbar;
}

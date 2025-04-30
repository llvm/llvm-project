extern __thread int tlsvar __attribute__((tls_model("initial-exec")));

void *
in_dso (void)
{
  return &tlsvar;
}

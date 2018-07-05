extern void foo(int n);
extern void __gcov_flush(void);

int bar1 = 0;
int bar2 = 1;

void bar(int n) {
  if (n % 5 == 0)
    bar1++;
  else
    bar2++;
}

int main(int argc, char *argv[]) {
#ifdef SHARED_CALL_BEFORE_GCOV_FLUSH
  foo(1);
#endif

  bar(5);

  __gcov_flush();

  bar(5);

#ifdef SHARED_CALL_AFTER_GCOV_FLUSH
  foo(1);
#endif

#ifdef EXIT_ABRUPTLY
  _exit(0);
#endif

  bar(5);

  return 0;
}

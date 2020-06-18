extern void __gcov_flush();
extern int remove(const char *);
int main(void) {
  __gcov_flush();

  if (remove("instrprof-gcov-__gcov_flush-multiple.gcda") != 0) {
    return 1;
  }

  __gcov_flush();
  __gcov_flush();

  if (remove("instrprof-gcov-__gcov_flush-multiple.gcda") != 0) {
    return 1;
  }

  return 0;
}

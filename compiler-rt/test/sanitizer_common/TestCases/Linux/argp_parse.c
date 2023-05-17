// RUN: %clang %s -o %t && %run %t -o baz

// argp_parse is glibc specific.
// UNSUPPORTED: android, target={{.*(freebsd|netbsd).*}}

#include <argp.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

struct test {
  const char *option_value;
};

static const struct argp_option options[] = {
    {"option", 'o', "OPTION", 0, "Option", 0},
    {NULL, 0, NULL, 0, NULL, 0},
};

static error_t parser(int key, char *arg, struct argp_state *state) {
  if (key == 'o') {
    ((struct test *)(state->input))->option_value = arg;
    return 0;
  }
  return ARGP_ERR_UNKNOWN;
}

static struct argp argp = {.options = options, .parser = parser};

void test_nulls(char *argv0) {
  char *argv[] = {argv0, NULL};
  int res = argp_parse(NULL, 1, argv, 0, NULL, NULL);
  assert(res == 0);
}

void test_synthetic(char *argv0) {
  char *argv[] = {argv0, "-o", "foo", "bar", NULL};
  struct test t = {NULL};
  int arg_index;
  int res = argp_parse(&argp, 4, argv, 0, &arg_index, &t);
  assert(res == 0);
  assert(arg_index == 3);
  assert(strcmp(t.option_value, "foo") == 0);
}

void test_real(int argc, char **argv) {
  struct test t = {NULL};
  int arg_index;
  int res = argp_parse(&argp, argc, argv, 0, &arg_index, &t);
  assert(res == 0);
  assert(arg_index == 3);
  assert(strcmp(t.option_value, "baz") == 0);
}

int main(int argc, char **argv) {
  test_nulls(argv[0]);
  test_synthetic(argv[0]);
  test_real(argc, argv);
  return EXIT_SUCCESS;
}

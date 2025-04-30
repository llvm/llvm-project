#include <argp.h>
#include <stdio.h>
#include <stdlib.h>

static struct argp_option argp_options[] = {
  { "dstaddr", 'd', "ADDR", 0,
    "set destination (peer) address to ADDR" },
  { "peer", 'p', "ADDR", OPTION_ALIAS },
  { NULL }
};

static error_t parse_opt (int key, char *arg, struct argp_state *state);

static struct argp argp =
{
  argp_options, parse_opt
};

static int cnt;

static int
do_test (int argc, char *argv[])
{
  int remaining;
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);
  return cnt != 4;
}

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
  {
  case 'd':
  case 'p':
    printf ("got '%c' with argument '%s'\n", key, arg);
    ++cnt;
    break;
  case 0:
  case ARGP_KEY_END:
  case ARGP_KEY_NO_ARGS:
  case ARGP_KEY_INIT:
  case ARGP_KEY_SUCCESS:
  case ARGP_KEY_FINI:
    // Ignore.
    return ARGP_ERR_UNKNOWN;
  default:
    printf ("invalid key '%x'\n", key);
    exit (1);
  }
  return 0;
}

#define TEST_FUNCTION do_test (argc, argv)
#include "../test-skeleton.c"

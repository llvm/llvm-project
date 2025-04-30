static void do_prepare (void);
#define PREPARE(argc, argv) do_prepare ()
static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"


static char *fname;


static void
do_prepare (void)
{
  if (create_temp_file ("tst-getopt_long1", &fname) < 0)
    {
      printf ("cannot create temp file: %m\n");
      exit (1);
    }
}


static const struct option opts[] =
  {
    { "one", no_argument, NULL, '1' },
    { "two", no_argument, NULL, '2' },
    { "one-one", no_argument, NULL, '3' },
    { "four", no_argument, NULL, '4' },
    { "onto", no_argument, NULL, '5' },
    { NULL, 0, NULL, 0 }
  };


static int
do_test (void)
{
  if (freopen (fname, "w+", stderr) == NULL)
    {
      printf ("freopen failed: %m\n");
      return 1;
    }

  char *argv[] = { (char *) "program", (char *) "--on" };
  int argc = 2;

  int c = getopt_long (argc, argv, "12345", opts, NULL);
  printf ("return value: %c\n", c);

  rewind (stderr);
  char *line = NULL;
  size_t len = 0;
  if (getline (&line, &len, stderr) < 0)
    {
      printf ("cannot read stderr redirect: %m\n");
      return 1;
    }
  printf ("message = \"%s\"\n", line);

  static const char expected[] = "\
program: option '--on' is ambiguous; possibilities: '--one' '--one-one' '--onto'\n";

  return c != '?' || strcmp (line, expected) != 0;
}

/* BZ #5451 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <support/temp_file.h>

static char *templ_filename;

// Writes template given as parameter to file,
// specified as the argument
static void
output_to_template_file (const char *str)
{
  FILE *fd = fopen (templ_filename, "w");
  if (fd == NULL)
    {
      printf ("Can not open file for writing\n");
      exit (1);
    }

  fprintf (fd, "%s\n", str);
  fclose (fd);
}

// Calls getdate() function with specified parameter,
// specified as the argument, also checks the contents of
// file with template and prints the result
static int
process_getdate_on (const char *str)
{
  struct tm *res;
  char templ[1000];
  FILE *fd = fopen (templ_filename, "r");

  if (fd == NULL)
    {
      printf ("Can not open file for reading\n");
      exit (1);
    }

  if (fgets (templ, 1000, fd) == NULL)
    {
      printf ("Can not read file\n");
      exit (1);
    }
  fclose (fd);

  res = getdate (str);
  if (res == NULL)
    {
      printf ("Failed on getdate(\"%s\"), template is: %s", str, templ);
      printf ("Error number: %d\n\n", getdate_err);
      return 1;
    }
  printf ("Success on getdate(\"%s\"), template is: %s\n", str, templ);
  printf ("Result is\n");
  printf ("Seconds: %d\n", res->tm_sec);
  printf ("Minutes: %d\n", res->tm_min);
  printf ("Hour: %d\n", res->tm_hour);
  printf ("Day of month: %d\n", res->tm_mday);
  printf ("Month of year: %d\n", res->tm_mon);
  printf ("Years since 1900: %d\n", res->tm_year);
  printf ("Day of week: %d\n", res->tm_wday);
  printf ("Day of year: %d\n", res->tm_yday);
  printf ("Daylight Savings flag: %d\n\n", res->tm_isdst);
  return 0;
}

static int
do_test (int argc, char *argv[])
{

  templ_filename = argv[1];

  setenv ("DATEMSK", templ_filename, 1);

  /*
   * The following 4 testcases reproduce the problem:
   * 1. Templates "%S" and "%M" are not processed,
   *    when used without "%H" template
   */
  int res = 0;
  output_to_template_file ("%M");
  res |= process_getdate_on ("1");

  output_to_template_file ("%M %H");
  res |= process_getdate_on ("1 2");

  output_to_template_file ("%S");
  res |= process_getdate_on ("1");

  output_to_template_file ("%S %H");
  res |= process_getdate_on ("1 2");

  /*
   * The following 9 testcases reproduce the problem:
   * 2. Templates "%Y", "%y", "%d", "%C", "%C %y"
   *    are not processed separately
   */
  output_to_template_file ("%Y");
  process_getdate_on ("2001");

  output_to_template_file ("%Y %m");
  res |= process_getdate_on ("2001 3");

  output_to_template_file ("%y");
  res |= process_getdate_on ("70");

  output_to_template_file ("%y %m");
  res |= process_getdate_on ("70 3");

  output_to_template_file ("%d");
  res |= process_getdate_on ("06");

  output_to_template_file ("%d %m");
  res |= process_getdate_on ("25 3");

  output_to_template_file ("%C");
  res |= process_getdate_on ("20");

  output_to_template_file ("%C %y %m");
  res |= process_getdate_on ("20 3 2");

  output_to_template_file ("%C %y");
  res |= process_getdate_on ("20 5");

  /*
   * The following testcase reproduces the problem:
   * 3. When template is "%Y %m", day of month is not set
   *    to 1 as standard requires
   */
  output_to_template_file ("%Y %m");
  res |= process_getdate_on ("2008 3");

  return res;
}
#define TEST_FUNCTION_ARGV do_test

static void
do_prepare (int argc, char **argv)
{
  if (argc < 2)
    {
      puts ("Command line: progname template_filename_full_path");
      exit (1);
    }
  add_temp_file (argv[1]);
}
#define PREPARE do_prepare

#include <support/test-driver.c>

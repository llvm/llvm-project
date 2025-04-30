#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int
main (int argc, char **argv)
{
  static const struct option options[] =
    {
      {"required", required_argument, NULL, 'r'},
      {"optional", optional_argument, NULL, 'o'},
      {"none",     no_argument,       NULL, 'n'},
      {"color",    no_argument,       NULL, 'C'},
      {"colour",   no_argument,       NULL, 'C'},
      {NULL,       0,                 NULL, 0 }
    };

  int aflag = 0;
  int bflag = 0;
  char *cvalue = NULL;
  int Cflag = 0;
  int nflag = 0;
  int index;
  int c;
  int result = 0;

  while ((c = getopt_long (argc, argv, "abc:", options, NULL)) >= 0)
    switch (c)
      {
      case 'a':
	aflag = 1;
	break;
      case 'b':
	bflag = 1;
	break;
      case 'c':
	cvalue = optarg;
	break;
      case 'C':
	++Cflag;
	break;
      case '?':
	fputs ("Unknown option.\n", stderr);
	return 1;
      default:
	fprintf (stderr, "This should never happen!\n");
	return 1;

      case 'r':
	printf ("--required %s\n", optarg);
	result |= strcmp (optarg, "foobar") != 0;
	break;
      case 'o':
	printf ("--optional %s\n", optarg);
	result |= optarg == NULL || strcmp (optarg, "bazbug") != 0;
	break;
      case 'n':
	puts ("--none");
	nflag = 1;
	break;
      }

  printf ("aflag = %d, bflag = %d, cvalue = %s, Cflags = %d, nflag = %d\n",
	  aflag, bflag, cvalue, Cflag, nflag);

  result |= (aflag != 1 || bflag != 1 || cvalue == NULL
	     || strcmp (cvalue, "foobar") != 0 || Cflag != 3 || nflag != 1);

  for (index = optind; index < argc; index++)
    printf ("Non-option argument %s\n", argv[index]);

  result |= optind + 1 != argc || strcmp (argv[optind], "random") != 0;

  return result;
}

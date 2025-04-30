#include <ftw.h>
#include <getopt.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>


int do_depth;
int do_chdir;
int do_phys;
int do_exit;
char *skip_subtree;
char *skip_siblings;

struct option options[] =
{
  { "depth", no_argument, &do_depth, 1 },
  { "chdir", no_argument, &do_chdir, 1 },
  { "phys", no_argument, &do_phys, 1 },
  { "skip-subtree", required_argument, NULL, 't' },
  { "skip-siblings", required_argument, NULL, 's' },
  { "early-exit", no_argument, &do_exit, 1 },
  { NULL, 0, NULL, 0 }
};

const char *flag2name[] =
{
  [FTW_F] = "FTW_F",
  [FTW_D] = "FTW_D",
  [FTW_DNR] = "FTW_DNR",
  [FTW_NS] = "FTW_NS",
  [FTW_SL] = "FTW_SL",
  [FTW_DP] = "FTW_DP",
  [FTW_SLN] = "FTW_SLN"
};


static int
cb (const char *name, const struct stat *st, int flag, struct FTW *f)
{
  if (do_exit && strcmp (name + f->base, "file@2"))
    return FTW_CONTINUE;

  printf ("base = \"%.*s\", file = \"%s\", flag = %s",
	  f->base, name, name + f->base, flag2name[flag]);
  if (do_chdir)
    {
      char *cwd = getcwd (NULL, 0);
      printf (", cwd = %s", cwd);
      free (cwd);
    }
  printf (", level = %d\n", f->level);

  if (skip_siblings && strcmp (name + f->base, skip_siblings) == 0)
    return FTW_SKIP_SIBLINGS;

  if (skip_subtree && strcmp (name + f->base, skip_subtree) == 0)
    return FTW_SKIP_SUBTREE;

  return do_exit ? 26 : FTW_CONTINUE;
}

int
main (int argc, char *argv[])
{
  int opt;
  int r;
  int flag = 0;
  mtrace ();

  while ((opt = getopt_long_only (argc, argv, "", options, NULL)) != -1)
    {
      if (opt == 't')
        skip_subtree = optarg;
      else if (opt == 's')
        skip_siblings = optarg;
    }

  if (do_chdir)
    flag |= FTW_CHDIR;
  if (do_depth)
    flag |= FTW_DEPTH;
  if (do_phys)
    flag |= FTW_PHYS;
  if (skip_subtree || skip_siblings)
    {
      flag |= FTW_ACTIONRETVAL;
      if (do_exit)
	{
	  printf ("--early-exit cannot be used together with --skip-{siblings,subtree}");
	  exit (1);
	}
    }

  char *cw1 = getcwd (NULL, 0);

  r = nftw (optind < argc ? argv[optind] : ".", cb, do_exit ? 1 : 3, flag);
  if (r < 0)
    perror ("nftw");

  char *cw2 = getcwd (NULL, 0);

  if (strcmp (cw1, cw2) != 0)
    {
      printf ("current working directory before and after nftw call differ:\n"
	      "before: %s\n"
	      "after:  %s\n", cw1, cw2);
      exit (1);
    }

  if (do_exit)
    {
      puts (r == 26 ? "succeeded" : "failed");
      return r == 26 ? 0 : 1;
    }
  return r;
}

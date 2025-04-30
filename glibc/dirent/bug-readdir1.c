#include <dirent.h>
#include <errno.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>


int
main (void)
{
  DIR *dirp;
  struct dirent* ent;

  /* open a dir stream */
  dirp = opendir ("/tmp");
  if (dirp == NULL)
    {
      if (errno == ENOENT)
	exit (0);

      perror ("opendir");
      exit (1);
    }

  /* close the directory file descriptor, making it invalid */
  if (close (dirfd (dirp)) != 0)
    {
      puts ("could not close directory file descriptor");
      /* This is not an error.  It is not guaranteed this is possible.  */
      return 0;
    }

  ent = readdir (dirp);

  return ent != NULL || errno != EBADF;
}

#include <grp.h>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

int
main (int argc, char *argv[])
{
  uid_t me;
  struct passwd *my_passwd;
  struct group *my_group = NULL;
  char **members;

  me = getuid ();
  my_passwd = getpwuid (me);
  if (my_passwd == NULL)
    printf ("Cannot find user entry for UID %d\n", me);
  else
    {
      printf ("My login name is %s.\n", my_passwd->pw_name);
      printf ("My uid is %d.\n", (int)(my_passwd->pw_uid));
      printf ("My home directory is %s.\n", my_passwd->pw_dir);
      printf ("My default shell is %s.\n", my_passwd->pw_shell);

      my_group = getgrgid (my_passwd->pw_gid);
      if (my_group == NULL)
	printf ("No data for group %d found\n", my_passwd->pw_gid);
      else
	{
	  printf ("My default group is %s (%d).\n",
		  my_group->gr_name, (int)(my_passwd->pw_gid));
	  printf ("The members of this group are:\n");
	  for (members = my_group->gr_mem; *members != NULL; ++members)
	    printf ("  %s\n", *members);
	}
    }

  return my_passwd && my_group ? EXIT_SUCCESS : EXIT_FAILURE;
}

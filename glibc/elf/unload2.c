#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <error.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>

#define MAPS ((struct link_map *) _r_debug.r_map)

#define OUT \
  for (map = MAPS; map != NULL; map = map->l_next)			      \
    if (map->l_type == lt_loaded)					      \
      printf ("name = \"%s\", direct_opencount = %d\n",			      \
	      map->l_name, (int) map->l_direct_opencount);		      \
  fflush (stdout)

int
main (void)
{
  void *h[3];
  struct link_map *map;
  void (*fp) (void);

  h[0] = dlopen ("unload2mod.so", RTLD_LAZY);
  h[1] = dlopen ("unload2mod.so", RTLD_LAZY);
  if (h[0] == NULL || h[1] == NULL)
    error (EXIT_FAILURE, errno, "cannot load \"unload2mod.so\"");
  h[2] = dlopen ("unload2dep.so", RTLD_LAZY);
  if (h[2] == NULL)
    error (EXIT_FAILURE, errno, "cannot load \"unload2dep.so\"");

  puts ("\nAfter loading everything:");
  OUT;

  dlclose (h[0]);

  puts ("\nAfter unloading \"unload2mod.so\" once:");
  OUT;

  dlclose (h[1]);

  puts ("\nAfter unloading \"unload2mod.so\" twice:");
  OUT;

  fp = dlsym (h[2], "foo");
  puts ("\nnow calling `foo'");
  fflush (stdout);
  fp ();
  puts ("managed to call `foo'");
  fflush (stdout);

  dlclose (h[2]);

  puts ("\nAfter unloading \"unload2dep.so\":");
  OUT;

  return 0;
}

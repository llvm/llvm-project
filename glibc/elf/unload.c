/* Test for unloading (really unmapping) of objects.  By Franz Sirl.
   This test does not have to passed in all dlopen() et.al. implementation
   since it is not required the unloading actually happens.  But we
   require it for glibc.  */

#include <dlfcn.h>
#include <link.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>

#define MAPS ((struct link_map *) _r_debug.r_map)

#define OUT \
  for (map = MAPS; map != NULL; map = map->l_next)			      \
    if (map->l_type == lt_loaded)					      \
      printf ("name = \"%s\", direct_opencount = %d\n",			      \
	      map->l_name, (int) map->l_direct_opencount);		      \
  fflush (stdout)

typedef struct
{
  void *next;
} strct;

int
main (void)
{
   void *sohandle;
   strct *testdat;
   int ret;
   int result = 0;
   struct link_map *map;

   mtrace ();

   puts ("\nBefore");
   OUT;

   sohandle = dlopen ("unloadmod.so", RTLD_NOW | RTLD_GLOBAL);
   if (sohandle == NULL)
     {
       printf ("*** first dlopen failed: %s\n", dlerror ());
       exit (1);
     }

   puts ("\nAfter loading unloadmod.so");
   OUT;

   testdat = dlsym (sohandle, "testdat");
   testdat->next = (void *) -1;

   ret = dlclose (sohandle);
   if (ret != 0)
     {
       puts ("*** first dlclose failed");
       result = 1;
     }

   puts ("\nAfter closing unloadmod.so");
   OUT;

   sohandle = dlopen ("unloadmod.so", RTLD_NOW | RTLD_GLOBAL);
   if (sohandle == NULL)
     {
       printf ("*** second dlopen failed: %s\n", dlerror ());
       exit (1);
     }

   puts ("\nAfter loading unloadmod.so the second time");
   OUT;

   testdat = dlsym (sohandle, "testdat");
   if (testdat->next == (void *) -1)
     {
       puts ("*** testdat->next == (void *) -1");
       result = 1;
     }

   ret = dlclose (sohandle);
   if (ret != 0)
     {
       puts ("*** second dlclose failed");
       result = 1;
     }

   puts ("\nAfter closing unloadmod.so again");
   OUT;

   return result;
}

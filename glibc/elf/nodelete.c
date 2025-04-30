#include <dlfcn.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>


static sigjmp_buf jmpbuf;


int fini_ran;


static void
__attribute__ ((noreturn))
handler (int sig)
{
  siglongjmp (jmpbuf, 1);
}


static int
do_test (void)
{
  /* We are testing the two possibilities to mark an object as not deletable:
     - marked on the linker commandline with `-z nodelete'
     - with the RTLD_NODELETE flag at dlopen()-time.

     The test we are performing should be safe.  We are loading the objects,
     get the address of variables in the respective object, unload the object
     and then try to read the variable.  If the object is unloaded this
     should lead to an segmentation fault.  */
  int result = 0;
  void *p;
  struct sigaction sa;

  sa.sa_handler = handler;
  sigfillset (&sa.sa_mask);
  sa.sa_flags = SA_RESTART;

  if (sigaction (SIGSEGV, &sa, NULL) == -1)
    printf ("cannot install signal handler: %m\n");

  p = dlopen ("nodelmod1.so", RTLD_LAZY);
  if (p == NULL)
    {
      printf ("failed to load \"nodelmod1.so\": %s\n", dlerror ());
      result = 1;
    }
  else
    {
      int *varp;

      puts ("succeeded loading \"nodelmod1.so\"");

      varp = dlsym (p, "var1");
      if (varp == NULL)
	{
	  puts ("failed to get address of \"var1\" in \"nodelmod1.so\"");
	  result = 1;
	}
      else
	{
	  *varp = 20000720;

	  /* Now close the object.  */
	  fini_ran = 0;
	  if (dlclose (p) != 0)
	    {
	      puts ("failed to close \"nodelmod1.so\"");
	      result = 1;
	    }
	  else if (! sigsetjmp (jmpbuf, 1))
	    {
	      /* Access the variable again.  */
	      if (*varp != 20000720)
		{
		  puts ("\"var1\" value not correct");
		  result = 1;
		}
	      else if (fini_ran != 0)
		{
		  puts ("destructor of \"nodelmod1.so\" ran");
		  result = 1;
		}
	      else
		puts ("-z nodelete test succeeded");
	    }
	  else
	    {
	      /* We caught an segmentation fault.  */
	      puts ("\"nodelmod1.so\" got deleted");
	      result = 1;
	    }
	}
    }

  p = dlopen ("nodelmod2.so", RTLD_LAZY | RTLD_NODELETE);
  if (p == NULL)
    {
      printf ("failed to load \"nodelmod2.so\": %s\n", dlerror ());
      result = 1;
    }
  else
    {
      int *varp;

      puts ("succeeded loading \"nodelmod2.so\"");

      varp = dlsym (p, "var2");
      if (varp == NULL)
	{
	  puts ("failed to get address of \"var2\" in \"nodelmod2.so\"");
	  result = 1;
	}
      else
	{
	  *varp = 42;

	  /* Now close the object.  */
	  fini_ran = 0;
	  if (dlclose (p) != 0)
	    {
	      puts ("failed to close \"nodelmod2.so\"");
	      result = 1;
	    }
	  else if (! sigsetjmp (jmpbuf, 1))
	    {
	      /* Access the variable again.  */
	      if (*varp != 42)
		{
		  puts ("\"var2\" value not correct");
		  result = 1;
		}
	      else if (fini_ran != 0)
		{
		  puts ("destructor of \"nodelmod2.so\" ran");
		  result = 1;
		}
	      else
		puts ("RTLD_NODELETE test succeeded");
	    }
	  else
	    {
	      /* We caught an segmentation fault.  */
	      puts ("\"nodelmod2.so\" got deleted");
	      result = 1;
	    }
	}
    }

  p = dlopen ("nodelmod3.so", RTLD_LAZY);
  if (p == NULL)
    {
      printf ("failed to load \"nodelmod3.so\": %s\n", dlerror ());
      result = 1;
    }
  else
    {
      int *(*fctp) (void);

      puts ("succeeded loading \"nodelmod3.so\"");

      fctp = dlsym (p, "addr");
      if (fctp == NULL)
	{
	  puts ("failed to get address of \"addr\" in \"nodelmod3.so\"");
	  result = 1;
	}
      else
	{
	  int *varp = fctp ();

	  *varp = -1;

	  /* Now close the object.  */
	  fini_ran = 0;
	  if (dlclose (p) != 0)
	    {
	      puts ("failed to close \"nodelmod3.so\"");
	      result = 1;
	    }
	  else if (! sigsetjmp (jmpbuf, 1))
	    {
	      /* Access the variable again.  */
	      if (*varp != -1)
		{
		  puts ("\"var_in_mod4\" value not correct");
		  result = 1;
		}
	      else if (fini_ran != 0)
		{
		  puts ("destructor of \"nodelmod4.so\" ran");
		  result = 1;
		}
	      else
		puts ("-z nodelete in dependency succeeded");
	    }
	  else
	    {
	      /* We caught an segmentation fault.  */
	      puts ("\"nodelmod4.so\" got deleted");
	      result = 1;
	    }
	}
    }

  return result;
}

#include <support/test-driver.c>

#include <iostream>

/* makedb needs selinux dso's.  */
#ifdef HAVE_SELINUX
# include <selinux/selinux.h>
#endif

using namespace std;

/* The purpose of this file is to indicate to the build system which
   shared objects need to be copied into the testroot, such as gcc or
   selinux support libraries.  This program is never executed, only
   scanned for dependencies on shared objects, so the code below may
   seem weird - it's written to survive gcc optimization and force
   such dependencies.
*/

int
main (int argc, char **argv)
{
  /* Complexity to keep gcc from optimizing this away.  */
  cout << (argc > 1 ? argv[1] : "null");
#ifdef HAVE_SELINUX
  /* This exists to force libselinux.so to be required.  */
  cout << "selinux " << is_selinux_enabled ();
#endif
  return 0;
}

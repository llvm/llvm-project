/* Test module for making nonexecutable stacks executable
   on load of a DSO that requires executable stacks.  */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void callme (void (*callback) (void));

/* This is a function that makes use of executable stack by
   using a local function trampoline.  */
void
tryme (void)
{
  bool ok = false;
  void callback (void) { ok = true; }

  callme (&callback);

  if (ok)
    printf ("DSO called ok (local %p, trampoline %p)\n", &ok, &callback);
  else
    abort ();
}

void
callme (void (*callback) (void))
{
  (*callback) ();
}

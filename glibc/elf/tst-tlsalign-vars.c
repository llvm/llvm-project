/* This is for tst-tlsalign-extern.c, which see.  It's essential for the
   purpose of the test that these definitions be in a separate translation
   unit from the code using the variables.  */

__thread int tdata1 = 1;
__thread int tdata2 __attribute__ ((aligned (0x10))) = 2;
__thread int tdata3 __attribute__ ((aligned (0x1000))) = 4;
__thread int tbss1;
__thread int tbss2 __attribute__ ((aligned (0x10)));
__thread int tbss3 __attribute__ ((aligned (0x1000)));

/* This function is never called.  But its presence in this translation
   unit makes GCC emit the variables above in the order defined (perhaps
   because it's the order in which they're used here?) rather than
   reordering them into descending order of alignment requirement--and so
   keeps it more similar to the tst-tlsalign-static.c case--just in case
   that affects the bug (though there is no evidence that it does).  */

void
unused (void)
{
  tdata1 = -1;
  tdata2 = -2;
  tdata3 = -3;
  tbss1 = -4;
  tbss2 = -5;
  tbss3 = -6;
}

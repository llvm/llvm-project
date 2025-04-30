/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_mbrlen.c
 *
 *	 MBRLEN:  size_t mbrlen (const char *s, size_t n, mbstate_t *ps);
 */

/*
 *  NOTE:
 *	  (1) A mbstate object is initialized for
 *	      every new data record by the test program.
 *
 *	  (2) USE_MBCURMAX is defined as a value of 99.
 *
 */


TST_MBRLEN tst_mbrlen_loc [] = {
  {
    { Tmbrlen, TST_LOC_de },
    {
      { /*----------------- #01 -----------------*/
	{
	  {
	    { 1, "",	   0,		   0, 0 },
	    { 1, "",	   1,		   0, 0 },
	    { 1, "\300",	   USE_MBCURMAX,   0, 0 },
	  }
	},
	{
	  {
	    { 0,		1,  -2,		     },
	    { 0,		1,  0,		     },
	    { 0,		1,  1,		     },
	  }
	}
      },
      { /*----------------- #02 -----------------*/
	{
	  {
	    { 1, "\300\001",   0,		   0, 0 },
	    { 1, "\300\001",   1,		   0, 0 },
	    { 1, "\317\001",   USE_MBCURMAX,   0, 0 },
	  }
	},
	{
	  {
	    { 0,		1,  -2,		     },
	    { 0,		1,  1,		     },
	    { 0,		1,  1,		     },
	  }
	}
      },
      { .is_last = 1 }
    }
  },
  {
    { Tmbrlen, TST_LOC_enUS },
    {
      { /*----------------- #01 -----------------*/
	{
	  {
	    { 1, "A",	   0,		   0, 0 },
	    { 1, "A",	   1,		   0, 0 },
	    { 1, "A",	   USE_MBCURMAX,   0, 0 },
	  }
	},
	{
	  {
	    { 0,		1,  -2,		     },
	    { 0,		1,  1,		     },
	    { 0,		1,  1,		     },
	  }
	}
      },
      { /*----------------- #02 -----------------*/
	{
	  {
	    { 1, "\317\001",   0,		   1, 0 },
	    { 1, "\317\001",   1,		   1, 0 },
	    { 1, "\317\001",   USE_MBCURMAX,   1, 0 },
	  }
	},
	{
	  {
	    { 0,		1,  -2,		     },
	    { EILSEQ,	1, -1,		     },
	    { EILSEQ,	1, -1,		     },
	  }
	}
      },
      { .is_last = 1 }
    }
  },
  {
    { Tmbrlen, TST_LOC_eucJP },
    {
      { /*----------------- #01 -----------------*/
	{
	  {
	    { 1, "\317\302",   1,		   1, 1 },
	    { 0, "",	       0,		   1, 0 },
	    { 1, "\317\302",   USE_MBCURMAX,	   1, 1 },
	  }
	},
	{
	  {
	    { 0,		1, -2,		     },
	    { 0,		1, -1,		     },
	    { 0,		1,  2,		     },
	  }
	}
      },
      { /*----------------- #02 -----------------*/
	{
	  {
	    { 1, "\317",	   1,		   1, 0 },
	    { 1, "\302",	   1,		   1, 0 },
	    { 1, "\317\302",   USE_MBCURMAX,   0, 0 },
	  }
	},
	{
	  {
	    { 0,		1, -2,		     },
	    /* XXX ISO C explicitly says that the return value does not
	       XXX reflect the bytes contained in the state.  */
	    { 0,		1, +1,		     },
	    { 0,		1,  2,		     },
	  }
	}
      },
      { /*----------------- #03 -----------------*/
	{
	  {
	    { 1, "\216\217",   0,		   0, 0 },
	    { 1, "\216\217",   1,		   0, 0 },
	    { 1, "\216\217",   USE_MBCURMAX,   0, 0 },
	  }
	},
	{
	  {
	    { 0,		1,  -2,		     },
	    { 0,		1, -2,		     },
	    { EILSEQ,	1, -1,		     },
	  }
	}
      },
      { .is_last = 1 }
    }
  },
  {
    { Tmbrlen, TST_LOC_end }
  }
};

/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN CLIBRARY
 *
 *	 FILE:	dat_wcstod.c
 *
 *	 WCSTOD:  double wcstod (const wchar_t *np, wchar_t **endp);
 */


/*
 *  NOTE:
 *	  need more test data!
 *
 */


TST_WCSTOD tst_wcstod_loc [] = {
  {
    { Twcstod, TST_LOC_de },
    {
      {
	/*01*/
	/*I*/
	{{ 0x0030,0x0030,0x0030,0x002C,0x0030,0x0030,0x0030,0x0030,0x0000 }},
	/*E*/
	{ 0,1,0.0,	       0.0,				  0x0000   }
      },
      {
	/*02*/
	/*I*/
	{{ 0x0031,0x0032,0x0033,0x002C,0x0034,0x0035,0x0036,0x0040,0x0000 }},
	/*E*/
	{ 0,1,123.456,	       123.456,			   0x0040  }
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcstod, TST_LOC_enUS },
    {
      {
	/*01*/
	/*I*/
	{{ 0x0030,0x0030,0x0030,0x002E,0x0030,0x0030,0x0030,0x0030,0x0000 }},
	/*E*/
	{ 0,1,0.0,	       0.0,				  0x0000   }
      },
      {
	/*02*/
	/*I*/
	{{ 0x0031,0x0032,0x0033,0x002E,0x0034,0x0035,0x0036,0x0040,0x0000 }},
	/*E*/
	{ 0,1,123.456,	       123.456,			   0x0040  }
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcstod, TST_LOC_eucJP },
    {
      {
	/*01*/
	/*I*/
	{{ 0x0031,0x0032,0x0033,0x002E,0x0034,0x0035,0x0036,0x0040,0x0000 }},
	/*E*/
	{ 0,1,123.456,	       123.456,			   0x0040  }
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcstod, TST_LOC_end }
  }
};

/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN CLIBRARY
 *
 *	 FILE:	dat_wctype.c
 *
 *	 WCTYPE:  wctype_t  wctype( const char *class );
 */

/*
 *  NOTE:
 *	  When a return value is expected to be 0 (false),
 *	  set ret_flg=1 and set ret_val=0.
 *	  Otherwise just set ret_flg=0.
 */


TST_WCTYPE tst_wctype_loc [] = {

    {	{ Twctype, TST_LOC_de },
	{
	  { /*inp*/ { "alnum"	       },  /* #01 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "alpha"	       },  /* #02 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "cntrl"	       },  /* #03 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "digit"	       },  /* #04 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "graph"	       },  /* #05 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "lower"	       },  /* #06 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "print"	       },  /* #07 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "punct"	       },  /* #08 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "space"	       },  /* #09 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "upper"	       },  /* #10 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "xdigit"	       },  /* #11 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { ""	       },  /* #12 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "ideograph"      },  /* #13 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "english"	       },  /* #14 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "ascii"	       },  /* #15 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "special"	       },  /* #16 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twctype, TST_LOC_enUS },
	{
	  { /*inp*/ { "alnum"	       },  /* #01 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "alpha"	       },  /* #02 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "cntrl"	       },  /* #03 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "digit"	       },  /* #04 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "graph"	       },  /* #05 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "lower"	       },  /* #06 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "print"	       },  /* #07 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "punct"	       },  /* #08 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "space"	       },  /* #09 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "upper"	       },  /* #10 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "xdigit"	       },  /* #11 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { ""	       },  /* #12 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "ideograph"      },  /* #13 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "english"	       },  /* #14 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "ascii"	       },  /* #15 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "special"	       },  /* #16 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twctype, TST_LOC_eucJP },
	{
	  { /*inp*/ { "alnum"	       },  /* #01 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "alpha"	       },  /* #02 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "cntrl"	       },  /* #03 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "digit"	       },  /* #04 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "graph"	       },  /* #05 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "lower"	       },  /* #06 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "print"	       },  /* #07 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "punct"	       },  /* #08 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "space"	       },  /* #09 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "upper"	       },  /* #10 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "xdigit"	       },  /* #11 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "ideogram"       },  /* #12 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "phonogram"      },  /* #13 */
	    /*exp*/ { 0,1,0,	       },
	  },
	  { /*inp*/ { "jspace"	       },  /* #14 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "jhira"	       },  /* #15 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "jkata"	       },  /* #16 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "jkanji"	       },  /* #17 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { /*inp*/ { "jdigit"	       },  /* #18 */
	    /*exp*/ { 0,0,0,	       },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twctype, TST_LOC_end }}
};

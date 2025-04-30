/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_wcwidth.c
 *
 *	 WCWIDTH:  int wcwidth (wchar_t wc);
 */

TST_WCWIDTH tst_wcwidth_loc [] = {
    {
      { Twcwidth, TST_LOC_de },
      {
	{ /*inp*/ { 0x0000		     },	 /* #01 */
	  /*exp*/ { 0,	1,0,	     },
	},
	{ /*inp*/ { 0x0020		     },	 /* #02 */
	  /*exp*/ { 0,	1,1,	     },
	},
	{ /*inp*/ { 0x007F		     },	 /* #03 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	{ /*inp*/ { 0x0080		     },	 /* #04 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	{ /*inp*/ { 0x00A1		     },	 /* #05 */
	  /*exp*/ { 0,	1,1,	     },
	},
	{ /*inp*/ { 0x00C1		     },	 /* #06 */
	  /*exp*/ { 0,	1,1,	     },
	},
	{ /*inp*/ { 0x3041		     },	 /* #07 */
	  /*exp*/ { 0,	1,EOF,	     },
	},
	{ .is_last = 1 }
      }
    },
    {
      { Twcwidth, TST_LOC_enUS },
      {
	{ /*inp*/ { 0x0000		     },	 /* #01 */
	  /*exp*/ { 0,	1,0,	     },
	},
	{ /*inp*/ { 0x0020		     },	 /* #02 */
	  /*exp*/ { 0,	1,1,	     },
	},
	{ /*inp*/ { 0x007F		     },	 /* #03 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	{ /*inp*/ { 0x0080		     },	 /* #04 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	{ /*inp*/ { 0x00A1		     },	 /* #05 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	{ /*inp*/ { 0x00C1		     },	 /* #06 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	{ /*inp*/ { 0x3041		     },	 /* #07 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	{ .is_last = 1 }
      }
    },
    {
      { Twcwidth, TST_LOC_eucJP },
      {
	{ /*inp*/ { 0x0000		     },	 /* #01 */
	  /*exp*/ { 0,	1,0,	     },
	},
	{ /*inp*/ { 0x0020		     },	 /* #02 */
	  /*exp*/ { 0,	1,1,	     },
	},
	{ /*inp*/ { 0x007F		     },	 /* #03 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	{ /*inp*/ { 0x0080		     },	 /* #04 */
	  /*exp*/ { 0,	1,-1,	     },
	},
	/* XXX U00A1 is a valid character in EUC-JP.  */
	{ /*inp*/ { 0x00A1		     },	 /* #05 */
	  /*exp*/ { 0,	1,2,	     },
	},
	/* jisx0212 */
	{ /*inp*/ { 0x00C1		     },	 /* #06 */
	  /*exp*/ { 0,	1,2,	     },
	},
	{ /*inp*/ { 0x3041		     },	 /* #07 */
	  /*exp*/ { 0,	1,2,	     },
	},
	{ .is_last = 1 }
      }
    },
    {
      { Twcwidth, TST_LOC_end }
    }
};

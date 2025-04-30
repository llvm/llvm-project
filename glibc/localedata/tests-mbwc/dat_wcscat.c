/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_wcscat.c
 *
 *	 WCSCAT:  wchar_t *wcscat (wchar_t *ws1, wchar_t *ws2)
 */

/* NOTE:
   Since this is not a locale sensitive function,
   it doesn't make sense to test the function on some
   locales. Better make different test cases for each locale ...
   (Also some wc* functions are not locale sensitive.)
*/


TST_WCSCAT tst_wcscat_loc [] = {

  {
    {Twcscat, TST_LOC_de},
    {
      /* 1 */
      {{{ 0x00C1,0x00C2,0x0000	},
	{			0x00C3,0x00C4,0x0000 }, },
       {   0,	0,    0,
	   { 0x00C1,0x00C2,0x00C3,0x00C4,0x0000 }	},
      },
      /* 2 */
      {{{ 0x0001,0x0002,0x0000	},
	{			0x0003,0x0004,0x0000 }, },
       {   0,	0,    0,
	   { 0x0001,0x0002,0x0003,0x0004,0x0000 }	},
      },
      /* 3 */
      {{{ 0x0000		  },
	{			0x00C3,0x00C4,0x0000 }, },
       {   0,	0,    0,
	   {		0x00C3,0x00C4,0x0000 }	},
      },
      /* 4 */
      {{{ 0x0001,0xFFFF,0x0000	},
	{			0x0080,0x0090,0x0000 }, },
       {   0,	0,    0,
	   { 0x0001,0xFFFF,0x0080,0x0090,0x0000 }	},
      },
      {.is_last = 1}
    }
  },
  {
    {Twcscat, TST_LOC_enUS},
    {
      /* 1 */
      {{{ 0x0041,0x0042,0x0000	},
	{		  0x0043,0x0044,0x0000 }, },
       {   0,	  0,	0,
	   { 0x0041,0x0042,0x0043,0x0044,0x0000 }  },
      },
      /* 2 */
      {{{ 0x0001,0x0002,0x0000	},
	{		  0x0003,0x0004,0x0000 }, },
       {   0,	  0,	0,
	   { 0x0001,0x0002,0x0003,0x0004,0x0000 }  },
      },
      /* 3 */
      {{{ 0x0000		    },
	{		  0x0043,0x0044,0x0000 }, },
       {   0,	  0,	0,
	   {		  0x0043,0x0044,0x0000 }  },
      },
      /* 4 */
      {{{ 0x0001,0xFFFF,0x0000	},
	{		  0x0080,0x0090,0x0000 }, },
       {   0,	  0,	0,
	   { 0x0001,0xFFFF,0x0080,0x0090,0x0000 }  },
      },
      {.is_last = 1}
    }
  },
  {
    {Twcscat, TST_LOC_eucJP},
    {
      /* 1 */
      {{{ 0x30A2,0x74E0,0x0000	},
	{			0xFF71,0x0041,0x0000 }, },
       {   0,	0,    0,
	   { 0x30A2,0x74E0,0xFF71,0x0041,0x0000 }	},
      },
      /* 2 */
      {{{ 0x0001,0x0002,0x0000	},
	{			0x0003,0x0004,0x0000 }, },
       {   0,	0,    0,
	   { 0x0001,0x0002,0x0003,0x0004,0x0000 }	},
      },
      /* 3 */
      {{{ 0x30A2,0xFF71,0x0000	},
	{			0x0000		     }, },
       {   0,	0,    0,
	   { 0x30A2,0xFF71,0x0000		     }	},
      },
      /* 4 */
      {{{ 0x0001,0xFFFF,0x0000	},
	{			0x0080,0x0090,0x0000 }, },
       {   0,	0,    0,
	   { 0x0001,0xFFFF,0x0080,0x0090,0x0000 }	},
      },
      {.is_last = 1}
    }
  },
  {
    {Twcscat, TST_LOC_end}
  }
};

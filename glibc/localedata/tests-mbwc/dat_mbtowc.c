/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_mbtowc.c
 *
 *	 MBTOWC:  int  mbtowc (wchar_t *wp, char *s, size_t n);
 */

/*  NOTE:
 *
 *	 int  mbtowc (wchar_t *wp, char *s, size_t n);
 *
 *	 where	     n: a maximum number of bytes
 *		return: the number of bytes
 *
 *
 *	  o When you feed a null pointer for a string (s) to the function,
 *	    set s_flg=0 instead of putting just a 'NULL' there.
 *	    Even if you put a 'NULL', it means a null string as well as "".
 *
 *	  o When s is a null pointer, the function checks state dependency.
 *
 *		state-dependent encoding      - return	NON-zero
 *		state-independent encoding    - return	0
 *
 *	    If state-dependent encoding is expected, set
 *
 *		s_flg = 0,  ret_flg = 0,  ret_val = +1
 *
 *	    If state-independent encoding is expected, set
 *
 *		s_flg = 0,  ret_flg = 0,  ret_val = 0
 *
 *
 *	    When you set ret_flg=1, the test program simply compares
 *	    an actual return value with an expected value. You can
 *	    check state-independent case (return value is 0) in that
 *	    way, but you can not check state-dependent case. So when
 *	    you check state- dependency in this test function:
 *	    tst_mbtowc(), set ret_flg=0 always. It's a special case
 *	    and the test function takes care of it.
 *
 *			  w_flg
 *			  |	s: (a null string; can't be (char *)NULL)
 *			  |	|
 *	       input.	{ 1, 0, (char)NULL, MB_LEN_MAX	},
 *			     |
 *			     s_flg=0: makes _s_ a null pointer.
 *
 *	       expect	{ 0,0,0,x,     0x0000	  },
 *			      | |
 *			      | ret_val: 0/+1
 *			      ret_flg=0
 *
 *
 *    Test data for State dependent encodings:
 *
 *	  mbtowc( NULL, NULL, 0 );	 ... first  data
 *	  mbtowc( &wc,	s1,  n1 );	 ... second data
 *	  mbtowc( &wc,	s2,  n2 );	 ... third  data
 * */

#include <limits.h>

TST_MBTOWC tst_mbtowc_loc [] = {
  {
    { Tmbtowc, TST_LOC_de },
    {
      { /*----------------- #01 -----------------*/
	{
	  {
	    { 1, 1, "\xfc\xe4\xf6",	    1	       },
	    { 1, 1, "\xfc\xe4\xf6",	    2	       },
	    { 1, 1, "\xfc\xe4\xf6",	    MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  1,   0x00FC },
	    { 0,  1,  1,   0x00FC },
	    { 0,  1,  1,   0x00FC },
	  }
	}
      },
      { /*----------------- #02 -----------------*/
	{
	  {
	    { 1, 1, "\177",	    MB_LEN_MAX },
	    { 1, 1, "\200",	    MB_LEN_MAX },
	    { 1, 1, "\201",	    MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  1,   0x007F },
	    { 0,  1,  1,   0x0080 },
	    { 0,  1,  1,   0x0081 },
	  }
	}
      },
      { /*----------------- #03 -----------------*/
	{
	  {
	    { 1, 1, "",			    MB_LEN_MAX },
	    { 0, 1, "\xfc\xe4\xf6",	    1	       },
	    { 0, 1, "\xfc\xe4\xf6",	    2	       },
	  }
	},
	{
	  {
	    { 0,  1,  0,   0x0000 },
	    { 0,  1,  1,   0x0000 },
	    { 0,  1,  1,   0x0000 },
	  }
	}
      },
      { /*----------------- #04 -----------------*/
	{
	  {
	    { 0, 1, "\xfc\xe4\xf6",	    MB_LEN_MAX },
	    { 0, 1, "\177",		    MB_LEN_MAX },
	    { 0, 1, "",	 		   MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  1,   0x0000 },
	    { 0,  1,  1,   0x0000 },
	    { 0,  1,  0,   0x0000 },
	  }
	}
      },
      { /*----------------- #05 -----------------*/
	{
	  {
	    { 0, 1, "\xfc\xe4\xf6",	MB_LEN_MAX },
	    { 0, 1, "\177",	   	MB_LEN_MAX },
	    { 0, 0, NULL, 		MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  1,   0x0000 },
	    { 0,  1,  1,   0x0000 },
	    { 0,  0,  0,   0x0000 },
	  }
	}
      },
      { .is_last = 1 }
    }
  },
  {
    { Tmbtowc, TST_LOC_enUS },
    {
      { /*----------------- #01 -----------------*/
	{
	  {
	    { 1, 1, "ABC",	    1	       },
	    { 1, 1, "ABC",	    2	       },
	    { 1, 1, "ABC",	    MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  1,   0x0041 },
	    { 0,  1,  1,   0x0041 },
	    { 0,  1,  1,   0x0041 },
	  }
	}
      },
      { /*----------------- #02 -----------------*/
	{
	  {
	    { 1, 1, "\177",	    MB_LEN_MAX },
	    { 1, 1, "\200",	    MB_LEN_MAX },
	    { 1, 1, "\201",	    MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  1,   0x007F },
	    { EILSEQ,  1, -1,   0x0000 },
	    { EILSEQ,  1, -1,   0x0000 },
	  }
	}
      },
      { /*----------------- #03 -----------------*/
	{
	  {
	    { 1, 1, "",	    MB_LEN_MAX },
	    { 0, 1, "ABC",	    1	       },
	    { 0, 1, "ABC",	    2	       },
	  }
	},
	{
	  {
	    { 0,  1,  0,   0x0000 },
	    { 0,  1,  1,   0x0000 },
	    { 0,  1,  1,   0x0000 },
	  }
	}
      },
      { /*----------------- #04 -----------------*/
	{
	  {
	    { 0, 1, "ABC",	    MB_LEN_MAX },
	    { 0, 1, "\177",	    MB_LEN_MAX },
	    { 0, 1, "",	    MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  1,   0x0000 },
	    { 0,  1,  1,   0x0000 },
	    { 0,  1,  0,   0x0000 },
	  }
	}
      },
      { /*----------------- #05 -----------------*/
	{
	  {
	    { 0, 1, "ABC",	    MB_LEN_MAX },
	    { 0, 1, "\177",	    MB_LEN_MAX },
	    { 0, 0, NULL,	    MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  1,   0x0000 },
	    { 0,  1,  1,   0x0000 },
	    { 0,  0,  0,   0x0000 },
	  }
	}
      },
      { .is_last = 1 }
    }
  },
  {
    { Tmbtowc, TST_LOC_eucJP },
    {
      { /*----------------- #01 -----------------*/
	{
	  {
	    { 1, 1, "\244\242A",      1          },
	    { 1, 1, "\244\242A",      2          },
	    { 1, 1, "\244\242A",      MB_LEN_MAX },
	  }
	},
	{
	  {
	    /* XXX EILSEQ was introduced in ISO C99.  */
	    { 0,	  1, -1,   0x0000 },
	    { 0,       1,  2,   0x3042 },
	    { 0,       1,  2,   0x3042 },
	  }
	}
      },
      { /*----------------- #02 -----------------*/
	{
	  {
	    { 1, 1, "\177\244\242",   MB_LEN_MAX },
	    { 1, 1, "\377\244\242",   MB_LEN_MAX },
	    { 1, 1, "\201\244\242",   MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1, +1,   0x007F },
	    { 0,  1, -1,   0x0000 },
	    { 0,  1, +1,   0x0081 },
	  }
	}
      },
      { /*----------------- #03 -----------------*/
	{
	  {
	    { 1, 1, "",         MB_LEN_MAX },
	    { 0, 1, "\244\242A",      1          },
	    { 0, 1, "\244\242A",      2          },
	  }
	},
	{
	  {
	    { 0,  1,  0,   0x0000 },
	    /* XXX EILSEQ was introduced in ISO C99.  */
	    { 0,       1, -1,   0x0000 },
	    { 0,  1,  2,   0x0000 },
	  }
	}
      },
      { /*----------------- #04 -----------------*/
	{
	  {
	    { 0, 1, "\244\242A",      MB_LEN_MAX },
	    { 0, 1, "\177\244\242",   MB_LEN_MAX },
	    { 0, 1, "",		      MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  2,   0x0000 },
	    { 0,  1, +1,   0x0000 },
	    { 0,  1,  0,   0x0000 },
	  }
	}
      },
      { /*----------------- #05 -----------------*/
	{
	  {
	    { 0, 1, "\244\242A",      MB_LEN_MAX },
	    { 0, 1, "\177\244\242",   MB_LEN_MAX },
	    { 0, 0, NULL,	      MB_LEN_MAX },
	  }
	},
	{
	  {
	    { 0,  1,  2,   0x0000 },
	    { 0,  1, +1,   0x0000 },
	    { 0,  0,  0,   0x0000 },
	  }
	}
      },
      { .is_last = 1 }
    }
  },
  {
    { Tmbtowc, TST_LOC_end }
  }
};

/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_wcscpy.c
 *
 *	 WCSCPY:  wchar_t *wcscpy (wchar_t *ws1, const wchar_t *ws2);
 */

TST_WCSCPY tst_wcscpy_loc [] = {

    {	{ Twcscpy, TST_LOC_de },
	{
	    { {		  { 0x00F1,0x00F2,0x00F3,0x0000	 }, },	   /* 1 */
	      {	 0,0,0,   { 0x00F1,0x00F2,0x00F3,0x0000, }  }, },
	    { {		  { 0x0000,0x00F2,0x00F3,0x0000	 }, },	   /* 2 */
	      {	 0,0,0,   { 0x0000,			 }  }, },
	    { .is_last = 1 }
	}
    },
    {	{ Twcscpy, TST_LOC_enUS },
	{
	    { {		  { 0x0041,0x0082,0x0043,0x0000	 }, },	   /* 1 */
	      {	 0,0,0,   { 0x0041,0x0082,0x0043,0x0000, }  }, },
	    { {		  { 0x0000,0x0082,0x0043,0x0000	 }, },	   /* 2 */
	      {	 0,0,0,   { 0x0000,			 }  }, },
	    { .is_last = 1 }
	}
    },
    {	{ Twcscpy, TST_LOC_eucJP },
	{
	    { {		  { 0x3041,0x0092,0x3043,0x0000	 }, },	   /* 1 */
	      {	 0,0,0,   { 0x3041,0x0092,0x3043,0x0000, }  }, },
	    { {		  { 0x0000,0x0092,0x3043,0x0000	 }, },	   /* 2 */
	      {	 0,0,0,   { 0x0000,			 }  }, },
	    { .is_last = 1 }
	}
    },
    {	{ Twcscpy, TST_LOC_end }}

};

/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_wctob.c
 *
 *	 ISW*:	int wctob( wint_t wc );
 */


TST_WCTOB tst_wctob_loc [] = {

    {	{ Twctob, TST_LOC_de },
	{
	  {  { WEOF   }, { 0,	 1, EOF	       }  },
	  {  { 0x0020 }, { 0,	 1, 0x20       }  },
	  {  { 0x0061 }, { 0,	 1, 0x61       }  },
	  {  { 0x0080 }, { 0,	 1, 0x80       }  },
	  {  { 0x00C4 }, { 0,	 1, 0xC4       }  },
	  {  { 0x30C4 }, { 0,	 1, EOF	       }  },
	  {  .is_last = 1 } /* Last element.  */
	}
    },
    {	{ Twctob, TST_LOC_enUS },
	{
	  {  { WEOF   }, { 0,	 1, EOF	       }  },
	  {  { 0x0020 }, { 0,	 1, 0x20       }  },
	  {  { 0x0061 }, { 0,	 1, 0x61       }  },
	  /* XXX These are no valid characters.  */
	  {  { 0x0080 }, { 0,	 1, EOF        }  },
	  {  { 0x00C4 }, { 0,	 1, EOF        }  },
	  {  { 0x30C4 }, { 0,	 1, EOF	       }  },
	  {  .is_last = 1 } /* Last element.  */
	}
    },
    {	{ Twctob, TST_LOC_eucJP },
	{
	  {  { WEOF   }, { 0,	 1, EOF	       }  },
	  {  { 0x0020 }, { 0,	 1, 0x20       }  },
	  {  { 0x0061 }, { 0,	 1, 0x61       }  },
	  {  { 0x0080 }, { 0,	 1, 0x80       }  },
	  {  { 0x00FF }, { 0,	 1, EOF        }  },
	  {  { 0x00C4 }, { 0,	 1, EOF	       }  },
	  {  { 0x30C4 }, { 0,	 1, EOF	       }  },
	  {  .is_last = 1 } /* Last element.  */
	}
    },
    {	{ Twctob, TST_LOC_end } }
};

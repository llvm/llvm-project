/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_iswupper.c
 *
 *	 ISW*:	int iswupper (wint_t wc);
 */


#include "dat_isw-funcs.h"


TST_ISW_LOC (UPPER, upper) = {

  {   TST_ISW_REC (de, upper)
      {
	{  { 0x0080 }, { 0,1,0 }  },	/* CTRL	    */
	{  { 0x009F }, { 0,1,0 }  },	/* CTRL	    */
	{  { 0x00A0 }, { 0,1,0 }  },	/* NB SPACE */
	{  { 0x00A1 }, { 0,1,0 }  },	/* UD !	    */
	{  { 0x00B0 }, { 0,1,0 }  },	/* Degree   */
	{  { 0x00B1 }, { 0,1,0 }  },	/* +- sign  */
	{  { 0x00B2 }, { 0,1,0 }  },	/* SUP 2    */
	{  { 0x00B3 }, { 0,1,0 }  },	/* SUP 3    */
	{  { 0x00B4 }, { 0,1,0 }  },	/* ACUTE    */
	{  { 0x00B8 }, { 0,1,0 }  },	/* CEDILLA  */
	{  { 0x00B9 }, { 0,1,0 }  },	/* SUP 1    */
	{  { 0x00BB }, { 0,1,0 }  },	/* >>	    */
	{  { 0x00BC }, { 0,1,0 }  },	/* 1/4	    */
	{  { 0x00BD }, { 0,1,0 }  },	/* 1/2	    */
	{  { 0x00BE }, { 0,1,0 }  },	/* 3/4	    */
	{  { 0x00BF }, { 0,1,0 }  },	/* UD ?	    */
	{  { 0x00C0 }, { 0,0,0 }  },	/* A Grave  */
	{  { 0x00D6 }, { 0,0,0 }  },	/* O dia    */
	{  { 0x00D7 }, { 0,1,0 }  },	/* multipl. */
	{  { 0x00D8 }, { 0,0,0 }  },	/* O stroke */
	{  { 0x00DF }, { 0,1,0 }  },	/* small Sh */
	{  { 0x00E0 }, { 0,1,0 }  },	/* a grave  */
	{  { 0x00F6 }, { 0,1,0 }  },	/* o dia    */
	{  { 0x00F7 }, { 0,1,0 }  },	/* division */
	{  { 0x00F8 }, { 0,1,0 }  },	/* o stroke */
	{  { 0x00FF }, { 0,1,0 }  },	/* y dia    */
	{ .is_last = 1 }		/* Last entry.	*/
      }
  },
  {   TST_ISW_REC (enUS, upper)
      {
	{  { WEOF   }, { 0,1,0 }  },
	{  { 0x0000 }, { 0,1,0 }  },
	{  { 0x001F }, { 0,1,0 }  },
	{  { 0x0020 }, { 0,1,0 }  },
	{  { 0x0021 }, { 0,1,0 }  },
	{  { 0x002F }, { 0,1,0 }  },
	{  { 0x0030 }, { 0,1,0 }  },
	{  { 0x0039 }, { 0,1,0 }  },
	{  { 0x003A }, { 0,1,0 }  },
	{  { 0x0040 }, { 0,1,0 }  },
	{  { 0x0041 }, { 0,0,0 }  },
	{  { 0x005A }, { 0,0,0 }  },
	{  { 0x005B }, { 0,1,0 }  },
	{  { 0x0060 }, { 0,1,0 }  },
	{  { 0x0061 }, { 0,1,0 }  },
	{  { 0x007A }, { 0,1,0 }  },
	{  { 0x007B }, { 0,1,0 }  },
	{  { 0x007E }, { 0,1,0 }  },
	{  { 0x007F }, { 0,1,0 }  },
	{  { 0x0080 }, { 0,1,0 }  },
	{ .is_last = 1 }		/* Last entry.  */
      }
  },
  {   TST_ISW_REC (eucJP, upper)
      {
	{  { 0x3041 }, { 0,1,0 }  },	/* HIRAGANA a	      */
	{  { 0x3094 }, { 0,1,0 }  },	/* HIRAGANA u"	      */
	{  { 0x30A1 }, { 0,1,0 }  },	/* KATAKANA a	      */
	{  { 0x30FA }, { 0,1,0 }  },	/* KATAKANA wo"	      */
	{  { 0xFF19 }, { 0,1,0 }  },	/* FULL 9	      */
	{  { 0xFF20 }, { 0,1,0 }  },	/* FULL @	      */
	{  { 0xFF3A }, { 0,0,0 }  },	/* FULL Z	      */
	{  { 0xFF40 }, { 0,1,0 }  },	/* FULL GRAVE ACC.    */
	{  { 0xFF5A }, { 0,1,0 }  },	/* FULL z	      */
	{  { 0xFF66 }, { 0,1,0 }  },	/* HALF KATA WO	      */
	{  { 0xFF6F }, { 0,1,0 }  },	/* HALF KATA tu	      */
	{  { 0xFF71 }, { 0,1,0 }  },	/* HALF KATA A	      */
	{  { 0xFF9E }, { 0,1,0 }  },	/* HALF KATA MI	      */
	{ .is_last = 1 }		/* Last entry.	*/
      }
  },
  {   TST_ISW_REC (end, upper) }
};

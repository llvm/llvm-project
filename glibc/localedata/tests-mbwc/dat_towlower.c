/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_towlower.c
 *
 *	 ISW*:	int towlower (wint_t wc);
 */


#include "dat_tow-funcs.h"


TST_TOW_LOC (LOWER, lower) = {

  {   TST_TOW_REC (de, lower)
      {
	{  { WEOF   }, { 0,  1, (wint_t)-1 }	},
	{  { 0x0080 }, { 0,  1, 0x0080     }	},
	{  { 0x00CC }, { 0,  1, 0x00EC     }	},
	{  { 0x00EC }, { 0,  1, 0x00EC     }	},
	{ .is_last = 1 } /* Last element.	 */
      }
  },
  {   TST_TOW_REC (enUS, lower)
      {
	{  { WEOF   }, { 0,  1, (wint_t)-1 }	},
	{  { 0x007F }, { 0,  1, 0x007F     }	},
	{  { 0x0041 }, { 0,  1, 0x0061     }	},
	{  { 0x0061 }, { 0,  1, 0x0061     }	},
	{ .is_last = 1 } /* Last element.	 */
      }
  },
  {   TST_TOW_REC (eucJP, lower)
      {
	{  { 0x007F }, { 0,  1, 0x007F     }	},
	{  { 0x0080 }, { 0,  1, 0x0080     }	},
	{  { 0xFF21 }, { 0,  1, 0xFF41     }	},
	{  { 0xFF41 }, { 0,  1, 0xFF41     }	},
	{ .is_last = 1 } /* Last element.	 */
      }
  },
  {   TST_TOW_REC (end, lower) }
};

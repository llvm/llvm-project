/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_towupper.c
 *
 *	 ISW*:	int towupper (wint_t wc);
 */


#include "dat_tow-funcs.h"


TST_TOW_LOC (UPPER, upper) = {

  {   TST_TOW_REC (de, upper)
      {
	{  { WEOF   }, { 0,  1, (wint_t)-1 }	},
	{  { 0x0080 }, { 0,  1, 0x0080     }	},
	{  { 0x00EC }, { 0,  1, 0x00CC     }	},
	{  { 0x00CC }, { 0,  1, 0x00CC     }	},
	{ .is_last = 1 } /* Last element.	 */
      }
  },
  {   TST_TOW_REC (enUS, upper)
      {
	{  { WEOF   }, { 0,  1, (wint_t)-1 }	},
	{  { 0x0080 }, { 0,  1, 0x0080     }	},
	{  { 0x0041 }, { 0,  1, 0x0041     }	},
	{  { 0x0061 }, { 0,  1, 0x0041     }	},
	{ .is_last = 1 } /* Last element.	 */
      }
  },
  {   TST_TOW_REC (eucJP, upper)
      {
	{  { WEOF   }, { 0,  1, (wint_t)-1 }	},
	{  { 0x007F }, { 0,  1, 0x007F     }	},
	{  { 0xFF41 }, { 0,  1, 0xFF21     }	},
	{  { 0xFF21 }, { 0,  1, 0xFF21     }	},
	{ .is_last = 1 } /* Last element.	 */
      }
  },
  {   TST_TOW_REC (end, upper) }
};

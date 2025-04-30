/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN CLIBRARY
 *
 *       FILE:  dat_wcslen.c
 *
 *       WCSLEN:  size_t wcslen (const wchar_t *ws);
 */


/*
 *  NOTE:
 *
 *      a header in each expected data:
 *
 *         int  err_val;  ... expected value for errno
 *        <typ> ret_flg; ... set ret_flg=1 to compare an expected
 *                           value with an actual value
 *        <typ> ret_val; ... expected value for return
 */


TST_WCSLEN tst_wcslen_loc [] = {

  {   { Twcslen, TST_LOC_de },
      {
	{ /*input.*/ { { 0x00D1,0x00D2,0x00D3,0x0000 } },  /* #01 */
	  /*expect*/ { 0,1,3,                        },
	},
	{ /*input.*/ { { 0x0000 }                      },  /* #02 */
	  /*expect*/ { 0,1,0,                        },
	},
	{ .is_last = 1 }
      }
  },
  {   { Twcslen, TST_LOC_enUS },
      {
	{ /*input.*/ { { 0x0041,0x0042,0x0043,0x0000 } },  /* #01 */
	  /*expect*/ { 0,1,3,                        },
	},
	{ /*input.*/ { { 0x0000 }                      },  /* #02 */
	  /*expect*/ { 0,1,0,                        },
	},
	{ .is_last = 1 }
      }
  },
  {   { Twcslen, TST_LOC_eucJP },
      {
	{ /*input.*/ { { 0x3041,0x3042,0x3043,0x0000 } },  /* #01 */
	  /*expect*/ { 0,1,3,                        },
	},
	{ /*input.*/ { { 0x0000 }                      },  /* #02 */
	  /*expect*/ { 0,1,0,                        },
	},
	{ .is_last = 1 }
      }
  },
  {   { Twcslen, TST_LOC_end }}
};

/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN CLIBRARY
 *
 *       FILE:  dat_wctrans.c
 *
 *       WCTRANS:  wctrans_t  wctrans( const char *charclass );
 */

/*
 *  NOTE:
 *        When a return value is expected to be 0 (false),
 *        set ret_flg=1 and set ret_val=0.
 *        Otherwise just set ret_flg=0.
 */


TST_WCTRANS tst_wctrans_loc [] = {

    {   { Twctrans, TST_LOC_de },
        {
          { /*inp*/ { ""               },  /* #1 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "upper"          },  /* #2 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "lower"          },  /* #3 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "toupper"        },  /* #4 */
            /*exp*/ { 0,0,0,         },
          },
          { /*inp*/ { "tolower"        },  /* #5 */
            /*exp*/ { 0,0,0,         },
          },
          { /*inp*/ { "xxxxx"          },  /* #6 */
            /*exp*/ { 0,1,0,         },
          },
	  { .is_last = 1 }
        }
    },
    {   { Twctrans, TST_LOC_enUS },
        {
          { /*inp*/ { ""               },  /* #1 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "upper"          },  /* #2 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "lower"          },  /* #3 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "toupper"        },  /* #4 */
            /*exp*/ { 0,0,0,         },
          },
          { /*inp*/ { "tolower"        },  /* #5 */
            /*exp*/ { 0,0,0,         },
          },
          { /*inp*/ { "xxxxx"          },  /* #6 */
            /*exp*/ { 0,1,0,         },
          },
	  { .is_last = 1 }
        }
    },
    {   { Twctrans, TST_LOC_eucJP },
        {
          { /*inp*/ { ""               },  /* #1 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "upper"          },  /* #2 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "lower"          },  /* #3 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "toupper"        },  /* #4 */
            /*exp*/ { 0,0,0,         },
          },
          { /*inp*/ { "tolower"        },  /* #5 */
            /*exp*/ { 0,0,0,         },
          },
          { /*inp*/ { "xxxxx"          },  /* #6 */
            /*exp*/ { 0,1,0,         },
          },
          { /*inp*/ { "tojhira"        },  /* #7 */
            /*exp*/ { 0,0,0,         },
          },
          { /*inp*/ { "tojkata"        },  /* #8 */
            /*exp*/ { 0,0,0,         },
          },
	  { .is_last = 1 }
        }
    },
    {   { Twctrans, TST_LOC_end }}
};

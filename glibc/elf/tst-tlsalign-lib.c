__thread int mod_tdata1 = 1;
__thread int mod_tdata2 __attribute__ ((aligned (0x10))) = 2;
__thread int mod_tdata3 __attribute__ ((aligned (0x1000))) = 4;
__thread int mod_tbss1;
__thread int mod_tbss2 __attribute__ ((aligned (0x10)));
__thread int mod_tbss3 __attribute__ ((aligned (0x1000)));

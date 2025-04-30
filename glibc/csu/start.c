/* This file should define the low-level program entry point,
   which should set up `__environ', and then do:
     __libc_init(argc, argv, __environ);
     exit(main(argc, argv, __environ));

   This file should be prepared to be the first thing in the text section (on
   Unix systems), or otherwise appropriately special.  */

/* The first piece of initialized data.  */
int __data_start = 0;
weak_alias (__data_start, data_start)

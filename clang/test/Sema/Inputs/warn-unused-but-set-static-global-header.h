// Header file for testing that header-defined static globals don't warn.
static int header_set_unused = 0;
static void header_init() { header_set_unused = 1; }

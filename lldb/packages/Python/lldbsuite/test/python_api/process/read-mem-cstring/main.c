#include <stdlib.h>
int main ()
{
   const char *empty_string = "";
   const char *one_letter_string = "1";
#if defined (__APPLE__)
   const char *invalid_memory_string = (char*)0x100; // lower 4k is always PAGEZERO & unreadable on darwin
#else
   const char *invalid_memory_string = -1ULL; // maybe an invalid address on other platforms?
#endif

   return empty_string[0] + one_letter_string[0]; // breakpoint here
}

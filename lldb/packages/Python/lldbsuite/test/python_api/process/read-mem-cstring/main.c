#include <stdlib.h>
int main ()
{
   const char *empty_string = "";
   const char *one_letter_string = "1";
   // This expects that lower 4k of memory will be mapped unreadable, which most
   // OSs do (to catch null pointer dereferences).
   const char *invalid_memory_string = (char*)0x100;

   return empty_string[0] + one_letter_string[0]; // breakpoint here
}

#include <stdio.h>
#include <wchar.h>

int
main (int argc, char *argv[])
{
  fputws (L"Hello world!\n", stdout);
  wprintf (L"This %s a %ls string: %d\n", "is", L"mixed", 42);
  wprintf (L"%Iu\n", 0xfeedbeef);
  return 0;
}

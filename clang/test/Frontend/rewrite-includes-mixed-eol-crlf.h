// Note: This header file has CRLF line endings.
// The indentation in some of the conditional inclusion directives below is
// intentional and is required for this test to function as a regression test
// for GH59736.
_Static_assert(__LINE__ == 5, "");
#if 1
_Static_assert(__LINE__ == 7, "");
  #if 1
  _Static_assert(__LINE__ == 9, "");
  #endif
#endif

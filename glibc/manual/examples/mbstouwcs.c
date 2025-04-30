#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

/* Do not include the above headers in the example.
*/
wchar_t *
mbstouwcs (const char *s)
{
  /* Include the null terminator in the conversion.  */
  size_t len = strlen (s) + 1;
  wchar_t *result = reallocarray (NULL, len, sizeof (wchar_t));
  if (result == NULL)
    return NULL;

  wchar_t *wcp = result;
  mbstate_t state;
  memset (&state, '\0', sizeof (state));

  while (true)
    {
      wchar_t wc;
      size_t nbytes = mbrtowc (&wc, s, len, &state);
      if (nbytes == 0)
        {
          /* Terminate the result string.  */
          *wcp = L'\0';
          break;
        }
      else if (nbytes == (size_t) -2)
        {
          /* Truncated input string.  */
          errno = EILSEQ;
          free (result);
          return NULL;
        }
      else if (nbytes == (size_t) -1)
        {
          /* Some other error (including EILSEQ).  */
          free (result);
          return NULL;
        }
      else
        {
          /* A character was converted.  */
          *wcp++ = towupper (wc);
          len -= nbytes;
          s += nbytes;
        }
    }
  return result;
}

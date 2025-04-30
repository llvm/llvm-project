/* Test assert_perror().
 *
 * This is hairier than you'd think, involving games with
 * stdio and signals.
 *
 */

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <setjmp.h>

jmp_buf rec;
char buf[160];

static void
sigabrt (int unused)
{
  longjmp (rec, 1);  /* recover control */
}

#undef NDEBUG
#include <assert.h>
static void
assert1 (void)
{
  assert_perror (1);
}

static void
assert2 (void)
{
  assert_perror (0);
}

#define NDEBUG
#include <assert.h>
static void
assert3 (void)
{
  assert_perror (2);
}

int
main(void)
{
  volatile int failed = 1;  /* safety in presence of longjmp() */

  fclose (stderr);
  stderr = tmpfile ();
  if (!stderr)
    abort ();

  signal (SIGABRT, sigabrt);

  if (!setjmp (rec))
    assert1 ();
  else
    failed = 0;  /* should happen */

  if (!setjmp (rec))
    assert2 ();
  else
    failed = 1; /* should not happen */

  if (!setjmp (rec))
    assert3 ();
  else
    failed = 1; /* should not happen */

  rewind (stderr);
  fgets (buf, 160, stderr);
  if (!strstr(buf, strerror (1)))
    failed = 1;

  fgets (buf, 160, stderr);
  if (strstr (buf, strerror (0)))
    failed = 1;

  fgets (buf, 160, stderr);
  if (strstr (buf, strerror (2)))
    failed = 1;

  return failed;
}

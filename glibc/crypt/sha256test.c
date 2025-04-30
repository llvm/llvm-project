#include <string.h>
#include "sha256.h"

static const struct
{
  const char *input;
  const char result[32];
} tests[] =
  {
    /* Test vectors from FIPS 180-2: appendix B.1.  */
    { "abc",
      "\xba\x78\x16\xbf\x8f\x01\xcf\xea\x41\x41\x40\xde\x5d\xae\x22\x23"
      "\xb0\x03\x61\xa3\x96\x17\x7a\x9c\xb4\x10\xff\x61\xf2\x00\x15\xad" },
    /* Test vectors from FIPS 180-2: appendix B.2.  */
    { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      "\x24\x8d\x6a\x61\xd2\x06\x38\xb8\xe5\xc0\x26\x93\x0c\x3e\x60\x39"
      "\xa3\x3c\xe4\x59\x64\xff\x21\x67\xf6\xec\xed\xd4\x19\xdb\x06\xc1" },
    /* Test vectors from the NESSIE project.  */
    { "",
      "\xe3\xb0\xc4\x42\x98\xfc\x1c\x14\x9a\xfb\xf4\xc8\x99\x6f\xb9\x24"
      "\x27\xae\x41\xe4\x64\x9b\x93\x4c\xa4\x95\x99\x1b\x78\x52\xb8\x55" },
    { "a",
      "\xca\x97\x81\x12\xca\x1b\xbd\xca\xfa\xc2\x31\xb3\x9a\x23\xdc\x4d"
      "\xa7\x86\xef\xf8\x14\x7c\x4e\x72\xb9\x80\x77\x85\xaf\xee\x48\xbb" },
    { "message digest",
      "\xf7\x84\x6f\x55\xcf\x23\xe1\x4e\xeb\xea\xb5\xb4\xe1\x55\x0c\xad"
      "\x5b\x50\x9e\x33\x48\xfb\xc4\xef\xa3\xa1\x41\x3d\x39\x3c\xb6\x50" },
    { "abcdefghijklmnopqrstuvwxyz",
      "\x71\xc4\x80\xdf\x93\xd6\xae\x2f\x1e\xfa\xd1\x44\x7c\x66\xc9\x52"
      "\x5e\x31\x62\x18\xcf\x51\xfc\x8d\x9e\xd8\x32\xf2\xda\xf1\x8b\x73" },
    { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      "\x24\x8d\x6a\x61\xd2\x06\x38\xb8\xe5\xc0\x26\x93\x0c\x3e\x60\x39"
      "\xa3\x3c\xe4\x59\x64\xff\x21\x67\xf6\xec\xed\xd4\x19\xdb\x06\xc1" },
    { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
      "\xdb\x4b\xfc\xbd\x4d\xa0\xcd\x85\xa6\x0c\x3c\x37\xd3\xfb\xd8\x80"
      "\x5c\x77\xf1\x5f\xc6\xb1\xfd\xfe\x61\x4e\xe0\xa7\xc8\xfd\xb4\xc0" },
    { "123456789012345678901234567890123456789012345678901234567890"
      "12345678901234567890",
      "\xf3\x71\xbc\x4a\x31\x1f\x2b\x00\x9e\xef\x95\x2d\xd8\x3c\xa8\x0e"
      "\x2b\x60\x02\x6c\x8e\x93\x55\x92\xd0\xf9\xc3\x08\x45\x3c\x81\x3e" }
  };


int
main (void)
{
  struct sha256_ctx ctx;
  char sum[32];
  int result = 0;
  int cnt;

  for (cnt = 0; cnt < (int) (sizeof (tests) / sizeof (tests[0])); ++cnt)
    {
      __sha256_init_ctx (&ctx);
      __sha256_process_bytes (tests[cnt].input, strlen (tests[cnt].input),
			      &ctx);
      __sha256_finish_ctx (&ctx, sum);
      if (memcmp (tests[cnt].result, sum, 32) != 0)
	{
	  printf ("test %d run %d failed\n", cnt, 1);
	  result = 1;
	}

      __sha256_init_ctx (&ctx);
      for (int i = 0; tests[cnt].input[i] != '\0'; ++i)
	__sha256_process_bytes (&tests[cnt].input[i], 1, &ctx);
      __sha256_finish_ctx (&ctx, sum);
      if (memcmp (tests[cnt].result, sum, 32) != 0)
	{
	  printf ("test %d run %d failed\n", cnt, 2);
	  result = 1;
	}
    }

  /* Test vector from FIPS 180-2: appendix B.3.  */
  char buf[1000];
  memset (buf, 'a', sizeof (buf));
  __sha256_init_ctx (&ctx);
  for (int i = 0; i < 1000; ++i)
    __sha256_process_bytes (buf, sizeof (buf), &ctx);
  __sha256_finish_ctx (&ctx, sum);
  static const char expected[32] =
    "\xcd\xc7\x6e\x5c\x99\x14\xfb\x92\x81\xa1\xc7\xe2\x84\xd7\x3e\x67"
    "\xf1\x80\x9a\x48\xa4\x97\x20\x0e\x04\x6d\x39\xcc\xc7\x11\x2c\xd0";
  if (memcmp (expected, sum, 32) != 0)
    {
      printf ("test %d failed\n", cnt++);
      result = 1;
    }

  __sha256_init_ctx (&ctx);
  for (int i = 0; i < 100000; ++i)
    __sha256_process_bytes (buf, 10, &ctx);
  __sha256_finish_ctx (&ctx, sum);
  if (memcmp (expected, sum, 32) != 0)
    {
      printf ("test %d failed\n", cnt++);
      result = 1;
    }

  return result;
}

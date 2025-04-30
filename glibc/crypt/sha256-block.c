#include <stdint.h>

/* Process LEN bytes of BUFFER, accumulating context into CTX.
   It is assumed that LEN % 64 == 0.  */
void
__sha256_process_block (const void *buffer, size_t len, struct sha256_ctx *ctx)
{
  const uint32_t *words = buffer;
  size_t nwords = len / sizeof (uint32_t);
  uint32_t a = ctx->H[0];
  uint32_t b = ctx->H[1];
  uint32_t c = ctx->H[2];
  uint32_t d = ctx->H[3];
  uint32_t e = ctx->H[4];
  uint32_t f = ctx->H[5];
  uint32_t g = ctx->H[6];
  uint32_t h = ctx->H[7];

  /* First increment the byte count.  FIPS 180-2 specifies the possible
     length of the file up to 2^64 bits.  Here we only compute the
     number of bytes.  */
  ctx->total64 += len;

  /* Process all bytes in the buffer with 64 bytes in each round of
     the loop.  */
  while (nwords > 0)
    {
      uint32_t W[64];
      uint32_t a_save = a;
      uint32_t b_save = b;
      uint32_t c_save = c;
      uint32_t d_save = d;
      uint32_t e_save = e;
      uint32_t f_save = f;
      uint32_t g_save = g;
      uint32_t h_save = h;

      /* Operators defined in FIPS 180-2:4.1.2.  */
#define Ch(x, y, z) ((x & y) ^ (~x & z))
#define Maj(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define S0(x) (CYCLIC (x, 2) ^ CYCLIC (x, 13) ^ CYCLIC (x, 22))
#define S1(x) (CYCLIC (x, 6) ^ CYCLIC (x, 11) ^ CYCLIC (x, 25))
#define R0(x) (CYCLIC (x, 7) ^ CYCLIC (x, 18) ^ (x >> 3))
#define R1(x) (CYCLIC (x, 17) ^ CYCLIC (x, 19) ^ (x >> 10))

      /* It is unfortunate that C does not provide an operator for
	 cyclic rotation.  Hope the C compiler is smart enough.  */
#define CYCLIC(w, s) ((w >> s) | (w << (32 - s)))

      /* Compute the message schedule according to FIPS 180-2:6.2.2 step 2.  */
      for (unsigned int t = 0; t < 16; ++t)
	{
	  W[t] = SWAP (*words);
	  ++words;
	}
      for (unsigned int t = 16; t < 64; ++t)
	W[t] = R1 (W[t - 2]) + W[t - 7] + R0 (W[t - 15]) + W[t - 16];

      /* The actual computation according to FIPS 180-2:6.2.2 step 3.  */
      for (unsigned int t = 0; t < 64; ++t)
	{
	  uint32_t T1 = h + S1 (e) + Ch (e, f, g) + K[t] + W[t];
	  uint32_t T2 = S0 (a) + Maj (a, b, c);
	  h = g;
	  g = f;
	  f = e;
	  e = d + T1;
	  d = c;
	  c = b;
	  b = a;
	  a = T1 + T2;
	}

      /* Add the starting values of the context according to FIPS 180-2:6.2.2
	 step 4.  */
      a += a_save;
      b += b_save;
      c += c_save;
      d += d_save;
      e += e_save;
      f += f_save;
      g += g_save;
      h += h_save;

      /* Prepare for the next round.  */
      nwords -= 16;
    }

  /* Put checksum in context given as argument.  */
  ctx->H[0] = a;
  ctx->H[1] = b;
  ctx->H[2] = c;
  ctx->H[3] = d;
  ctx->H[4] = e;
  ctx->H[5] = f;
  ctx->H[6] = g;
  ctx->H[7] = h;
}

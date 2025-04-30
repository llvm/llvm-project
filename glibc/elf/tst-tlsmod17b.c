#define P(N) extern int tlsmod17a##N (void);
#define PS P(0) P(1) P(2) P(3) P(4) P(5) P(6) P(7) P(8) P(9) \
	   P(10) P(12) P(13) P(14) P(15) P(16) P(17) P(18) P(19)
PS
#undef P

int
tlsmod17b (void)
{
  int res = 0;
#define P(N) res |= tlsmod17a##N ();
  PS
#undef P
  return res;
}

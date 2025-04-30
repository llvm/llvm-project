#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int next;

void
f00 (void)
{
  puts ("f00");
  if (next-- != 0)
    _exit (1);
}

void
f01 (void)
{
  puts ("f01");
  if (next-- != 1)
    _exit (1);
}

void
f02 (void)
{
  puts ("f02");
  if (next-- != 2)
    _exit (1);
}

void
f03 (void)
{
  puts ("f03");
  if (next-- != 3)
    _exit (1);
}

void
f04 (void)
{
  puts ("f04");
  if (next-- != 4)
    _exit (1);
}

void
f05 (void)
{
  puts ("f05");
  if (next-- != 5)
    _exit (1);
}

void
f06 (void)
{
  puts ("f06");
  if (next-- != 6)
    _exit (1);
}

void
f07 (void)
{
  puts ("f07");
  if (next-- != 7)
    _exit (1);
}

void
f08 (void)
{
  puts ("f08");
  if (next-- != 8)
    _exit (1);
}

void
f09 (void)
{
  puts ("f09");
  if (next-- != 9)
    _exit (1);
}

void
f10 (void)
{
  puts ("f10");
  if (next-- != 10)
    _exit (1);
}

void
f11 (void)
{
  puts ("f11");
  if (next-- != 11)
    _exit (1);
}

void
f12 (void)
{
  puts ("f12");
  if (next-- != 12)
    _exit (1);
}

void
f13 (void)
{
  puts ("f13");
  if (next-- != 13)
    _exit (1);
}

void
f14 (void)
{
  puts ("f14");
  if (next-- != 14)
    _exit (1);
}

void
f15 (void)
{
  puts ("f15");
  if (next-- != 15)
    _exit (1);
}

void
f16 (void)
{
  puts ("f16");
  if (next-- != 16)
    _exit (1);
}

void
f17 (void)
{
  puts ("f17");
  if (next-- != 17)
    _exit (1);
}

void
f18 (void)
{
  puts ("f18");
  if (next-- != 18)
    _exit (1);
}

void
f19 (void)
{
  puts ("f19");
  if (next-- != 19)
    _exit (1);
}

void
f20 (void)
{
  puts ("f20");
  if (next-- != 20)
    _exit (1);
}

void
f21 (void)
{
  puts ("f21");
  if (next-- != 21)
    _exit (1);
}

void
f22 (void)
{
  puts ("f22");
  if (next-- != 22)
    _exit (1);
}

void
f23 (void)
{
  puts ("f23");
  if (next-- != 23)
    _exit (1);
}

void
f24 (void)
{
  puts ("f24");
  if (next-- != 24)
    _exit (1);
}

void
f25 (void)
{
  puts ("f25");
  if (next-- != 25)
    _exit (1);
}

void
f26 (void)
{
  puts ("f26");
  if (next-- != 26)
    _exit (1);
}

void
f27 (void)
{
  puts ("f27");
  if (next-- != 27)
    _exit (1);
}

void
f28 (void)
{
  puts ("f28");
  if (next-- != 28)
    _exit (1);
}

void
f29 (void)
{
  puts ("f29");
  if (next-- != 29)
    _exit (1);
}

void
f30 (void)
{
  puts ("f30");
  if (next-- != 30)
    _exit (1);
}

void
f31 (void)
{
  puts ("f31");
  if (next-- != 31)
    _exit (1);
}

void
f32 (void)
{
  puts ("f32");
  if (next-- != 32)
    _exit (1);
}

void
f33 (void)
{
  puts ("f33");
  if (next-- != 33)
    _exit (1);
}

void
f34 (void)
{
  puts ("f34");
  if (next-- != 34)
    _exit (1);
}

void
f35 (void)
{
  puts ("f35");
  if (next-- != 35)
    _exit (1);
}

void
f36 (void)
{
  puts ("f36");
  if (next-- != 36)
    _exit (1);
}

void
f37 (void)
{
  puts ("f37");
  if (next-- != 37)
    _exit (1);
}

void
f38 (void)
{
  puts ("f38");
  if (next-- != 38)
    _exit (1);
}

void
f39 (void)
{
  puts ("f39");
  if (next-- != 39)
    _exit (1);
}

void
foo (void)
{
  atexit (f00);
  atexit (f01);
  atexit (f02);
  atexit (f03);
  atexit (f04);
  atexit (f05);
  atexit (f06);
  atexit (f07);
  atexit (f08);
  atexit (f09);

  atexit (f10);
  atexit (f11);
  atexit (f12);
  atexit (f13);
  atexit (f14);
  atexit (f15);
  atexit (f16);
  atexit (f17);
  atexit (f18);
  atexit (f19);

  atexit (f20);
  atexit (f21);
  atexit (f22);
  atexit (f23);
  atexit (f24);
  atexit (f25);
  atexit (f26);
  atexit (f27);
  atexit (f28);
  atexit (f29);

  atexit (f30);
  atexit (f31);
  atexit (f32);
  atexit (f33);
  atexit (f34);
  atexit (f35);
  atexit (f36);
  atexit (f37);
  atexit (f38);
  atexit (f39);

  next = 39;
}

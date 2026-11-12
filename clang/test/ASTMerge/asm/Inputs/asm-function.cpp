
unsigned char asmFunc(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned int bigres;
  unsigned char res;
  __asm__ ("0:\n1:\n" : [bigres] "=la"(bigres) : [la] "0"(la), [lb] "c"(lb) :
                        "edx", "cc");
  res = bigres;
  return res;
}

int asmFunc2(int i) {
  int res;
  asm ("mov %1, %0 \t\n"
       "inc %0 "
      : "=r" (res)
      : "r" (i)
      : "cc");
  return res;
}

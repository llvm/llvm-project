// Make sure we IR-interpret the expression correctly.

typedef unsigned int uint32_t;
struct S0 {
  signed f2;
};
static g_463 = 0x1561983AL;
void func_1(void)
{
  struct S0 l_19;
  l_19.f2 = 419;
  uint32_t l_4037 = 4294967295UL;
  l_19.f2 = g_463; //%self.expect("expr ((l_4037 % (-(g_463))) | l_19.f2)", substrs=['(unsigned int) $0 = 358717883'])
}
int main()
{
  func_1();
  return 0;
}

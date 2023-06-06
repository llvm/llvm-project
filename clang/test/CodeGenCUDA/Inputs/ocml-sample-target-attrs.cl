typedef unsigned long ulong;

__attribute__((target("gfx11-insts")))
ulong do_intrin_stuff(void)
{
  return __builtin_amdgcn_s_sendmsg_rtnl(0x0);
}

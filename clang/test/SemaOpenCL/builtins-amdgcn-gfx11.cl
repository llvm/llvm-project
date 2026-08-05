// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1030 -verify=GFX10 -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1100 -verify=GFX11 -S -o - %s

typedef unsigned int uint;
typedef unsigned long ulong;

void test(global uint* out1, global ulong* out2, int x) {
  *out1 = __builtin_amdgcn_s_sendmsg_rtn(0); // GFX10-error {{'__builtin_amdgcn_s_sendmsg_rtn' needs target feature gfx11-insts}}
  *out2 = __builtin_amdgcn_s_sendmsg_rtnl(0); // GFX10-error {{'__builtin_amdgcn_s_sendmsg_rtnl' needs target feature gfx11-insts}}
#if __has_builtin(__builtin_amdgcn_s_sendmsg_rtn)
  *out1 = __builtin_amdgcn_s_sendmsg_rtn(x); // GFX11-error {{argument to '__builtin_amdgcn_s_sendmsg_rtn' must be a constant integer}}
#endif
#if __has_builtin(__builtin_amdgcn_s_sendmsg_rtnl)
  *out2 = __builtin_amdgcn_s_sendmsg_rtnl(x); // GFX11-error {{argument to '__builtin_amdgcn_s_sendmsg_rtnl' must be a constant integer}}
#endif

  *out1 = __builtin_amdgcn_permlane64(x); // GFX10-error {{'__builtin_amdgcn_permlane64' needs target feature gfx11-insts}}
}

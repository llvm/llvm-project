#define __CLC_UPSAMPLE_DECL(BGENTYPE, GENTYPE, UGENTYPE)                       \
  _CLC_OVERLOAD _CLC_DECL BGENTYPE upsample(GENTYPE hi, UGENTYPE lo);

#define __CLC_UPSAMPLE_VEC(BGENTYPE, GENTYPE, UGENTYPE)                        \
  __CLC_UPSAMPLE_DECL(BGENTYPE, GENTYPE, UGENTYPE)                             \
  __CLC_UPSAMPLE_DECL(BGENTYPE##2, GENTYPE##2, UGENTYPE##2)                    \
  __CLC_UPSAMPLE_DECL(BGENTYPE##3, GENTYPE##3, UGENTYPE##3)                    \
  __CLC_UPSAMPLE_DECL(BGENTYPE##4, GENTYPE##4, UGENTYPE##4)                    \
  __CLC_UPSAMPLE_DECL(BGENTYPE##8, GENTYPE##8, UGENTYPE##8)                    \
  __CLC_UPSAMPLE_DECL(BGENTYPE##16, GENTYPE##16, UGENTYPE##16)

#define __CLC_UPSAMPLE_TYPES()                                                 \
  __CLC_UPSAMPLE_VEC(short, char, uchar)                                       \
  __CLC_UPSAMPLE_VEC(ushort, uchar, uchar)                                     \
  __CLC_UPSAMPLE_VEC(int, short, ushort)                                       \
  __CLC_UPSAMPLE_VEC(uint, ushort, ushort)                                     \
  __CLC_UPSAMPLE_VEC(long, int, uint)                                          \
  __CLC_UPSAMPLE_VEC(ulong, uint, uint)

__CLC_UPSAMPLE_TYPES()

#undef __CLC_UPSAMPLE_TYPES
#undef __CLC_UPSAMPLE_DECL
#undef __CLC_UPSAMPLE_VEC

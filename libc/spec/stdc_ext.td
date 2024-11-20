// Fixed point types.
// From ISO/IEC TR 18037:2008 standard:
// https://standards.iso.org/ittf/PubliclyAvailableStandards/c051126_ISO_IEC_TR_18037_2008.zip
def ShortFractType : NamedType<"short fract">;
def FractType : NamedType<"fract">;
def LongFractType : NamedType<"long fract">;
def UnsignedShortFractType : NamedType<"unsigned short fract">;
def UnsignedFractType : NamedType<"unsigned fract">;
def UnsignedLongFractType : NamedType<"unsigned long fract">;

def ShortAccumType : NamedType<"short accum">;
def AccumType : NamedType<"accum">;
def LongAccumType : NamedType<"long accum">;
def UnsignedShortAccumType : NamedType<"unsigned short accum">;
def UnsignedAccumType : NamedType<"unsigned accum">;
def UnsignedLongAccumType : NamedType<"unsigned long accum">;

def IntHrT : NamedType <"int_hr_t">;
def IntRT : NamedType<"int_r_t">;
def IntLrT : NamedType<"int_lr_t">;
def IntHkT : NamedType<"int_hk_t">;
def IntKT : NamedType<"int_k_t">;
def IntLkT : NamedType<"int_lk_t">;
def UIntUhrT : NamedType<"uint_uhr_t">;
def UIntUrT : NamedType<"uint_ur_t">;
def UIntUlrT : NamedType<"uint_ulr_t">;
def UIntUhkT : NamedType<"uint_uhk_t">;
def UIntUkT : NamedType<"uint_uk_t">;
def UIntUlkT : NamedType<"uint_ulk_t">;

def StdcExt : StandardSpec<"stdc_ext"> {
  // From ISO/IEC TR 18037:2008 standard:
  // https://standards.iso.org/ittf/PubliclyAvailableStandards/c051126_ISO_IEC_TR_18037_2008.zip
  HeaderSpec StdFix = HeaderSpec<
      "stdfix.h",
      [],  // macros
      [IntHrT,IntRT, IntLrT, IntHkT, IntKT, IntLkT, UIntUhrT, UIntUrT, UIntUlrT, UIntUhkT, UIntUkT, UIntUlkT],  // types
      [],  // enums
      [    // functions
          GuardedFunctionSpec<"abshr", RetValSpec<ShortFractType>, [ArgSpec<ShortFractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"absr", RetValSpec<FractType>, [ArgSpec<FractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"abslr", RetValSpec<LongFractType>, [ArgSpec<LongFractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"abshk", RetValSpec<ShortAccumType>, [ArgSpec<ShortAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"absk", RetValSpec<AccumType>, [ArgSpec<AccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"abslk", RetValSpec<LongAccumType>, [ArgSpec<LongAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"roundhr", RetValSpec<ShortFractType>, [ArgSpec<ShortFractType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"roundr", RetValSpec<FractType>, [ArgSpec<FractType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"roundlr", RetValSpec<LongFractType>, [ArgSpec<LongFractType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"roundhk", RetValSpec<ShortAccumType>, [ArgSpec<ShortAccumType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"roundk", RetValSpec<AccumType>, [ArgSpec<AccumType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"roundlk", RetValSpec<LongAccumType>, [ArgSpec<LongAccumType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"rounduhr", RetValSpec<UnsignedShortFractType>, [ArgSpec<UnsignedShortFractType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"roundur", RetValSpec<UnsignedFractType>, [ArgSpec<UnsignedFractType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"roundulr", RetValSpec<UnsignedLongFractType>, [ArgSpec<UnsignedLongFractType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"rounduhk", RetValSpec<UnsignedShortAccumType>, [ArgSpec<UnsignedShortAccumType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"rounduk", RetValSpec<UnsignedAccumType>, [ArgSpec<UnsignedAccumType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"roundulk", RetValSpec<UnsignedLongAccumType>, [ArgSpec<UnsignedLongAccumType>, ArgSpec<IntType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"hrbits", RetValSpec<ShortFractType>, [ArgSpec<IntHrT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"rbits", RetValSpec<FractType>, [ArgSpec<IntRT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"lrbits", RetValSpec<LongFractType>, [ArgSpec<IntLrT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"hkbits", RetValSpec<ShortAccumType>, [ArgSpec<IntHkT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"kbits", RetValSpec<AccumType>, [ArgSpec<IntKT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"lkbits", RetValSpec<LongAccumType>, [ArgSpec<IntLkT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"uhrbits", RetValSpec<UnsignedShortFractType>, [ArgSpec<UIntUhrT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"urbits", RetValSpec<UnsignedFractType>, [ArgSpec<UIntUrT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"ukbits", RetValSpec<UnsignedAccumType>, [ArgSpec<UIntUkT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"ulrbits", RetValSpec<UnsignedLongFractType>, [ArgSpec<UIntUlrT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"uhkbits", RetValSpec<UnsignedShortAccumType>, [ArgSpec<UIntUhkT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"ulkbits", RetValSpec<UnsignedLongAccumType>, [ArgSpec<UIntUlkT>], "LIBC_COMPILER_HAS_FIXED_POINT">,
      ]
  >;

  let Headers = [
    StdFix,
  ];
}

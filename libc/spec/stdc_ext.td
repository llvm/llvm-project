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

def StdcExt : StandardSpec<"stdc_ext"> {
  // From ISO/IEC TR 18037:2008 standard:
  // https://standards.iso.org/ittf/PubliclyAvailableStandards/c051126_ISO_IEC_TR_18037_2008.zip
  HeaderSpec StdFix = HeaderSpec<
      "stdfix.h",
      [],  // macros
      [],  // types
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

          GuardedFunctionSpec<"sqrtuhr", RetValSpec<UnsignedShortFractType>, [ArgSpec<UnsignedShortFractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"sqrtur", RetValSpec<UnsignedFractType>, [ArgSpec<UnsignedFractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"sqrtulr", RetValSpec<UnsignedLongFractType>, [ArgSpec<UnsignedLongFractType>], "LIBC_COMPILER_HAS_FIXED_POINT">,

          GuardedFunctionSpec<"sqrtuhk", RetValSpec<UnsignedShortAccumType>, [ArgSpec<UnsignedShortAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"sqrtuk", RetValSpec<UnsignedAccumType>, [ArgSpec<UnsignedAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
          GuardedFunctionSpec<"sqrtulk", RetValSpec<UnsignedLongAccumType>, [ArgSpec<UnsignedLongAccumType>], "LIBC_COMPILER_HAS_FIXED_POINT">,
      ]
  >;

  let Headers = [
    StdFix,
  ];
}

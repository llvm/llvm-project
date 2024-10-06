def StdC : StandardSpec<"stdc"> {

  NamedType StructTmType = NamedType<"struct tm">;
  PtrType StructTmPtr = PtrType<StructTmType>;
  PtrType TimeTTypePtr = PtrType<TimeTType>;
  NamedType ClockT = NamedType<"clock_t">;
  NamedType LocaleT = NamedType<"locale_t">;

  NamedType DivTType = NamedType<"div_t">;
  NamedType LDivTType = NamedType<"ldiv_t">;
  NamedType LLDivTType = NamedType<"lldiv_t">;

  NamedType JmpBuf = NamedType<"jmp_buf">;

  NamedType TssTType = NamedType<"tss_t">;
  PtrType TssTPtr = PtrType<TssTType>;
  NamedType TssDtorTType = NamedType<"tss_dtor_t">;

  HeaderSpec Assert = HeaderSpec<
      "assert.h",
      [
          Macro<"static_assert">,
          Macro<"assert">,
      ],
      [], // Types
      [], // Enumerations
      []
  >;

  FunctionAttrSpec ConstAttr = FunctionAttrSpec<"__LIBC_CONST_ATTR", [
    Cxx11FunctionAttr<"const", "gnu">,
    GnuFunctionAttr<"const">,
  ]>;

  HeaderSpec CType = HeaderSpec<
      "ctype.h",
      [], // Macros
      [
          LocaleT
      ], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "isalnum",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isalpha",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isblank",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "iscntrl",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isdigit",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isgraph",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "islower",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isprint",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "ispunct",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isspace",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isupper",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isxdigit",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "tolower",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "toupper",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "isalnum_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "isalpha_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "isblank_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "iscntrl_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "isdigit_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "isgraph_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "islower_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "isprint_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "ispunct_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "isspace_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "isupper_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "isxdigit_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "tolower_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "toupper_l",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<LocaleT>]
          >,
      ]
  >;

  NamedType FEnvT = NamedType<"fenv_t">;
  PtrType FEnvTPtr = PtrType<FEnvT>;
  ConstType ConstFEnvTPtr = ConstType<FEnvTPtr>;
  NamedType FExceptT = NamedType<"fexcept_t">;
  PtrType FExceptTPtr = PtrType<FExceptT>;
  ConstType ConstFExceptTPtr = ConstType<FExceptTPtr>;
  HeaderSpec Fenv = HeaderSpec<
      "fenv.h",
      [
          Macro<"FE_DIVBYZERO">,
          Macro<"FE_INEXACT">,
          Macro<"FE_INVALID">,
          Macro<"FE_OVERFLOW">,
          Macro<"FE_UNDERFLOW">,
          Macro<"FE_ALL_EXCEPT">,

          Macro<"FE_DOWNWARD">,
          Macro<"FE_TONEAREST">,
          Macro<"FE_TOWARDZERO">,
          Macro<"FE_UPWARD">,

          Macro<"FE_DFL_ENV">
      ],
      [
          FEnvT,
          FExceptT,
      ], // Types
      [], // Enumerations
      [
          FunctionSpec<
              "feclearexcept",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "fetestexcept",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "fetestexceptflag",
              RetValSpec<IntType>,
              [ArgSpec<ConstFExceptTPtr>, ArgSpec<IntType>]
          >,
          FunctionSpec<
              "feraiseexcept",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "fesetround",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "fegetround",
              RetValSpec<IntType>,
              []
          >,
          FunctionSpec<
              "fegetenv",
              RetValSpec<IntType>,
              [ArgSpec<FEnvTPtr>]
          >,
          FunctionSpec<
              "fesetenv",
              RetValSpec<IntType>,
              [ArgSpec<ConstFEnvTPtr>]
          >,
          FunctionSpec<
              "fegetexceptflag",
              RetValSpec<IntType>,
              [ArgSpec<FExceptTPtr>, ArgSpec<IntType>]
          >,
          FunctionSpec<
              "fesetexcept",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "fesetexceptflag",
              RetValSpec<IntType>,
              [ArgSpec<ConstFExceptTPtr>, ArgSpec<IntType>]
          >,
          FunctionSpec<
              "feholdexcept",
              RetValSpec<IntType>,
              [ArgSpec<FEnvTPtr>]
          >,
          FunctionSpec<
              "feupdateenv",
              RetValSpec<IntType>,
              [ArgSpec<ConstFEnvTPtr>]
          >,
      ]
  >;

  HeaderSpec String = HeaderSpec<
      "string.h",
      [
          Macro<"NULL">,
      ],
      [
          SizeTType,
      ],
      [], // Enumerations
      [
          FunctionSpec<
              "memcpy",
              RetValSpec<VoidPtr>,
              [ArgSpec<VoidRestrictedPtr>,
               ArgSpec<ConstVoidRestrictedPtr>,
               ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memmove",
              RetValSpec<VoidPtr>,
              [ArgSpec<VoidPtr>, ArgSpec<ConstVoidPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memcmp",
              RetValSpec<IntType>,
              [ArgSpec<ConstVoidPtr>, ArgSpec<ConstVoidPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memchr",
              RetValSpec<VoidPtr>,
              [ArgSpec<ConstVoidPtr>, ArgSpec<IntType>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memset",
              RetValSpec<VoidPtr>,
              [ArgSpec<VoidPtr>, ArgSpec<IntType>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "memset_explicit",
              RetValSpec<VoidPtr>,
              [ArgSpec<VoidPtr>, ArgSpec<IntType>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strcpy",
              RetValSpec<CharPtr>,
              [ArgSpec<CharRestrictedPtr>, ArgSpec<ConstCharRestrictedPtr>]
          >,
          FunctionSpec<
              "strncpy",
              RetValSpec<CharPtr>,
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strcat",
              RetValSpec<CharPtr>,
              [ArgSpec<CharRestrictedPtr>, ArgSpec<ConstCharRestrictedPtr>]
          >,
          FunctionSpec<
              "strncat",
              RetValSpec<CharPtr>,
              [ArgSpec<CharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strcmp",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strcoll",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strcoll_l",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "strncmp",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strxfrm",
              RetValSpec<SizeTType>,
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strxfrm_l",
              RetValSpec<SizeTType>,
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<SizeTType>,
               ArgSpec<LocaleT>]
          >,
          FunctionSpec<
              "strchr",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
          >,
          FunctionSpec<
              "strcspn",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strdup",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strndup",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>,ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "strpbrk",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strrchr",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>, ArgSpec<IntType>]
          >,
          FunctionSpec<
              "strspn",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strstr",
              RetValSpec<CharPtr>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "strtok",
              RetValSpec<CharPtr>,
              [ArgSpec<CharRestrictedPtr>, ArgSpec<ConstCharRestrictedPtr>]
          >,
          FunctionSpec<
              "strerror",
              RetValSpec<CharPtr>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "strlen",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstCharPtr>]
          >,
      ]
  >;

  HeaderSpec Math = HeaderSpec<
      "math.h",
      [
          Macro<"MATH_ERRNO">,
          Macro<"MATH_ERREXCEPT">,
          Macro<"math_errhandling">,

          Macro<"HUGE_VAL">,
          Macro<"INFINITY">,
          Macro<"NAN">,

          Macro<"FP_INT_UPWARD">,
          Macro<"FP_INT_DOWNWARD">,
          Macro<"FP_INT_TOWARDZERO">,
          Macro<"FP_INT_TONEARESTFROMZERO">,
          Macro<"FP_INT_TONEAREST">,

          Macro<"FP_ILOGB0">,
          Macro<"FP_ILOGBNAN">,

          Macro<"isfinite">,
          Macro<"isinf">,
          Macro<"isnan">,
      ],
      [
          NamedType<"float_t">,
          NamedType<"double_t">,
          NamedType<"float128">,
      ],
      [], // Enumerations
      [
          FunctionSpec<"cbrt", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"cbrtf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"copysign", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"copysignf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"copysignl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"copysignf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"copysignf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"ceil", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"ceilf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"ceill", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"ceilf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"ceilf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"daddl", RetValSpec<DoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          FunctionSpec<"ddivl", RetValSpec<DoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          FunctionSpec<"dfmal", RetValSpec<DoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          FunctionSpec<"dsubl", RetValSpec<DoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"fabs", RetValSpec<DoubleType>, [ArgSpec<DoubleType>], [ConstAttr]>,
          FunctionSpec<"fabsf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"fabsl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fabsf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fabsf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"fadd", RetValSpec<FloatType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"faddl", RetValSpec<FloatType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"fdim", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fdimf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fdiml", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fdimf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fdimf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"fdiv", RetValSpec<FloatType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fdivl", RetValSpec<FloatType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"ffma", RetValSpec<FloatType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"ffmal", RetValSpec<FloatType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"floor", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"floorf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"floorl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"floorf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"floorf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"fmin", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fminf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fminl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fminf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,
          GuardedFunctionSpec<"fminf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,

          FunctionSpec<"fmax", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmaxf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fmaxl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fmaxf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,
          GuardedFunctionSpec<"fmaxf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,

	  FunctionSpec<"fmaximum", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmaximumf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fmaximuml", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fmaximumf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fmaximumf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

	  FunctionSpec<"fmaximum_num", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmaximum_numf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fmaximum_numl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fmaximum_numf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fmaximum_numf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

	  FunctionSpec<"fmaximum_mag", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmaximum_magf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fmaximum_magl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fmaximum_magf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fmaximum_magf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

	  FunctionSpec<"fmaximum_mag_num", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmaximum_mag_numf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fmaximum_mag_numl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fmaximum_mag_numf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fmaximum_mag_numf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

	  FunctionSpec<"fminimum", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fminimumf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fminimuml", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fminimumf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fminimumf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

	  FunctionSpec<"fminimum_num", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fminimum_numf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fmaximum_numl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fminimum_numf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fminimum_numf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

	  FunctionSpec<"fminimum_mag", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fminimum_magf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fminimum_magl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fminimum_magf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fminimum_magf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

	  FunctionSpec<"fminimum_mag_num", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fminimum_mag_numf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fminimum_mag_numl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fminimum_mag_numf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fminimum_mag_numf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"fma", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmaf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>, ArgSpec<FloatType>]>,

          GuardedFunctionSpec<"f16fmaf128", RetValSpec<Float16Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128">,

          FunctionSpec<"fmod", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmodf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fmodl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"fmodf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fmodf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"frexp", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntPtr>]>,
          FunctionSpec<"frexpf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntPtr>]>,
          FunctionSpec<"frexpl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntPtr>]>,
          GuardedFunctionSpec<"frexpf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<IntPtr>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"frexpf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<IntPtr>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"fromfp", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"fromfpf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"fromfpl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          GuardedFunctionSpec<"fromfpf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fromfpf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"fromfpx", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"fromfpxf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"fromfpxl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          GuardedFunctionSpec<"fromfpxf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"fromfpxf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"fsub", RetValSpec<FloatType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fsubl", RetValSpec<FloatType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"ufromfp", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"ufromfpf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"ufromfpl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          GuardedFunctionSpec<"ufromfpf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"ufromfpf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"ufromfpx", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"ufromfpxf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"ufromfpxl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>]>,
          GuardedFunctionSpec<"ufromfpxf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"ufromfpxf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<IntType>, ArgSpec<UnsignedIntType>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"hypot", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"hypotf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,

          FunctionSpec<"ilogb", RetValSpec<IntType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"ilogbf", RetValSpec<IntType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"ilogbl", RetValSpec<IntType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"ilogbf16", RetValSpec<IntType>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"ilogbf128", RetValSpec<IntType>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"llogb", RetValSpec<LongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"llogbf", RetValSpec<LongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"llogbl", RetValSpec<LongType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"llogbf16", RetValSpec<LongType>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"llogbf128", RetValSpec<LongType>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"ldexp", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntType>]>,
          FunctionSpec<"ldexpf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntType>]>,
          FunctionSpec<"ldexpl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntType>]>,
          GuardedFunctionSpec<"ldexpf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<IntType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"ldexpf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<IntType>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"log10", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"log10f", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"log1p", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"log1pf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"log2", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"log2f", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"log", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"logf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"logb", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"logbf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"logbl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"logbf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"logbf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"modf", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoublePtr>]>,
          FunctionSpec<"modff", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatPtr>]>,
          FunctionSpec<"modfl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoublePtr>]>,
          GuardedFunctionSpec<"modff16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Ptr>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"modff128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Ptr>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"cos", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"cosf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"sin", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"sinf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"tan", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"tanf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"erff", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"exp", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"expf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          GuardedFunctionSpec<"expf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,

          FunctionSpec<"exp2", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"exp2f", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          GuardedFunctionSpec<"exp2f16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,

          FunctionSpec<"exp2m1f", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"expm1", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"expm1f", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          GuardedFunctionSpec<"expm1f16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,

          FunctionSpec<"exp10", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"exp10f", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          GuardedFunctionSpec<"exp10f16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,

          FunctionSpec<"remainder", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"remainderf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"remainderl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"remainderf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"remainderf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"remquo", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>, ArgSpec<IntPtr>]>,
          FunctionSpec<"remquof", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>, ArgSpec<IntPtr>]>,
          FunctionSpec<"remquol", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>, ArgSpec<IntPtr>]>,
          GuardedFunctionSpec<"remquof16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>, ArgSpec<IntPtr>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"remquof128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>, ArgSpec<IntPtr>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"round", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"roundf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"roundl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"roundf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"roundf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"roundeven", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"roundevenf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"roundevenl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"roundevenf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"roundevenf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"lround", RetValSpec<LongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"lroundf", RetValSpec<LongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"lroundl", RetValSpec<LongType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"lroundf16", RetValSpec<LongType>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"lroundf128", RetValSpec<LongType>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"llround", RetValSpec<LongLongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"llroundf", RetValSpec<LongLongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"llroundl", RetValSpec<LongLongType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"llroundf16", RetValSpec<LongLongType>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"llroundf128", RetValSpec<LongLongType>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"rint", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"rintf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"rintl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"rintf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"rintf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"lrint", RetValSpec<LongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"lrintf", RetValSpec<LongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"lrintl", RetValSpec<LongType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"lrintf16", RetValSpec<LongType>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"lrintf128", RetValSpec<LongType>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"llrint", RetValSpec<LongLongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"llrintf", RetValSpec<LongLongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"llrintl", RetValSpec<LongLongType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"llrintf16", RetValSpec<LongLongType>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"llrintf128", RetValSpec<LongLongType>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"sqrt", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"sqrtf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"sqrtl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"sqrtf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"trunc", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"truncf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"truncl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"truncf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"truncf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"nearbyint", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"nearbyintf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"nearbyintl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"nearbyintf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"nearbyintf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"nextafterf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"nextafter", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"nextafterl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"nextafterf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"nextafterf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"nexttowardf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<LongDoubleType>]>,
          FunctionSpec<"nexttoward", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<LongDoubleType>]>,
          FunctionSpec<"nexttowardl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"nexttowardf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,

          FunctionSpec<"nextdown", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"nextdownf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"nextdownl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"nextdownf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"nextdownf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"nextup", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"nextupf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"nextupl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"nextupf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"nextupf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"powf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"pow", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,

          FunctionSpec<"coshf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"sinhf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"tanhf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"acosf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"asinf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"asin", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,

          FunctionSpec<"atanf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"atan2", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"atan2f", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"atan2l", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"acoshf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"asinhf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"atanhf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"scalbln", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<LongType>]>,
          FunctionSpec<"scalblnf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<LongType>]>,
          FunctionSpec<"scalblnl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongType>]>,
          GuardedFunctionSpec<"scalblnf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<LongType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"scalblnf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<LongType>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"scalbn", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntType>]>,
          FunctionSpec<"scalbnf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntType>]>,
          FunctionSpec<"scalbnl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntType>]>,
          GuardedFunctionSpec<"scalbnf16", RetValSpec<Float16Type>, [ArgSpec<Float16Type>, ArgSpec<IntType>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"scalbnf128", RetValSpec<Float128Type>, [ArgSpec<Float128Type>, ArgSpec<IntType>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"nanf", RetValSpec<FloatType>, [ArgSpec<ConstCharPtr>]>,
          FunctionSpec<"nan", RetValSpec<DoubleType>, [ArgSpec<ConstCharPtr>]>,
          FunctionSpec<"nanl", RetValSpec<LongDoubleType>, [ArgSpec<ConstCharPtr>]>,
          GuardedFunctionSpec<"nanf16", RetValSpec<Float16Type>, [ArgSpec<ConstCharPtr>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"nanf128", RetValSpec<Float128Type>, [ArgSpec<ConstCharPtr>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"canonicalize", RetValSpec<IntType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"canonicalizef", RetValSpec<IntType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"canonicalizel", RetValSpec<IntType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"canonicalizef16", RetValSpec<IntType>, [ArgSpec<Float16Type>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"canonicalizef128", RetValSpec<IntType>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"dsqrtl", RetValSpec<DoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"totalorder", RetValSpec<IntType>, [ArgSpec<ConstDoublePtr>, ArgSpec<ConstDoublePtr>]>,
	  FunctionSpec<"totalorderf", RetValSpec<IntType>, [ArgSpec<ConstFloatPtr>, ArgSpec<ConstFloatPtr>]>,
	  FunctionSpec<"totalorderl", RetValSpec<IntType>, [ArgSpec<ConstLongDoublePtr>, ArgSpec<ConstLongDoublePtr>]>,
	  GuardedFunctionSpec<"totalorderf16", RetValSpec<IntType>, [ArgSpec<ConstFloat16Ptr>, ArgSpec<ConstFloat16Ptr>], "LIBC_TYPES_HAS_FLOAT16">,
	  GuardedFunctionSpec<"totalorderf128", RetValSpec<IntType>, [ArgSpec<ConstFloat128Ptr>, ArgSpec<ConstFloat128Ptr>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"totalordermag", RetValSpec<IntType>, [ArgSpec<ConstDoublePtr>, ArgSpec<ConstDoublePtr>]>,
          FunctionSpec<"totalordermagf", RetValSpec<IntType>, [ArgSpec<ConstFloatPtr>, ArgSpec<ConstFloatPtr>]>,
          FunctionSpec<"totalordermagl", RetValSpec<IntType>, [ArgSpec<ConstLongDoublePtr>, ArgSpec<ConstLongDoublePtr>]>,
          GuardedFunctionSpec<"totalordermagf16", RetValSpec<IntType>, [ArgSpec<ConstFloat16Ptr>, ArgSpec<ConstFloat16Ptr>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"totalordermagf128", RetValSpec<IntType>, [ArgSpec<ConstFloat128Ptr>, ArgSpec<ConstFloat128Ptr>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"getpayload", RetValSpec<DoubleType>, [ArgSpec<DoublePtr>]>,
	  FunctionSpec<"getpayloadf", RetValSpec<FloatType>, [ArgSpec<FloatPtr>]>,
          FunctionSpec<"getpayloadl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoublePtr>]>,
          GuardedFunctionSpec<"getpayloadf16", RetValSpec<Float16Type>, [ArgSpec<Float16Ptr>], "LIBC_TYPES_HAS_FLOAT16">,
          GuardedFunctionSpec<"getpayloadf128", RetValSpec<Float128Type>, [ArgSpec<Float128Ptr>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"setpayload", RetValSpec<IntType>, [ArgSpec<DoublePtr>, ArgSpec<DoubleType>]>,
          FunctionSpec<"setpayloadf", RetValSpec<IntType>, [ArgSpec<FloatPtr>, ArgSpec<FloatType>]>,
	  FunctionSpec<"setpayloadl", RetValSpec<IntType>, [ArgSpec<LongDoublePtr>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"setpayloadf16", RetValSpec<IntType>, [ArgSpec<Float16Ptr>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
	  GuardedFunctionSpec<"setpayloadf128", RetValSpec<IntType>, [ArgSpec<Float128Ptr>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          FunctionSpec<"setpayloadsig", RetValSpec<IntType>, [ArgSpec<DoublePtr>, ArgSpec<DoubleType>]>,
          FunctionSpec<"setpayloadsigf", RetValSpec<IntType>, [ArgSpec<FloatPtr>, ArgSpec<FloatType>]>,
	  FunctionSpec<"setpayloadsigl", RetValSpec<IntType>, [ArgSpec<LongDoublePtr>, ArgSpec<LongDoubleType>]>,
          GuardedFunctionSpec<"setpayloadsigf16", RetValSpec<IntType>, [ArgSpec<Float16Ptr>, ArgSpec<Float16Type>], "LIBC_TYPES_HAS_FLOAT16">,
	  GuardedFunctionSpec<"setpayloadsigf128", RetValSpec<IntType>, [ArgSpec<Float128Ptr>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT128">,

          GuardedFunctionSpec<"f16addf128", RetValSpec<Float16Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128">,

          GuardedFunctionSpec<"f16subf128", RetValSpec<Float16Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128">,

          FunctionSpec<"fmul", RetValSpec<FloatType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmull", RetValSpec<FloatType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"dmull", RetValSpec<DoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          GuardedFunctionSpec<"f16mulf128", RetValSpec<Float16Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128">,

          FunctionSpec<"fsqrt", RetValSpec<FloatType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"fsqrtl", RetValSpec<FloatType>, [ArgSpec<LongDoubleType>]>,

          GuardedFunctionSpec<"f16divf128", RetValSpec<Float16Type>, [ArgSpec<Float128Type>, ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128">,

          GuardedFunctionSpec<"f16sqrtf128", RetValSpec<Float16Type>, [ArgSpec<Float128Type>], "LIBC_TYPES_HAS_FLOAT16_AND_FLOAT128">,

          FunctionSpec<"lgamma", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"lgammaf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"lgammal", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,
      ]
  >;

  HeaderSpec StdIO = HeaderSpec<
      "stdio.h",
      [
          Macro<"stdin">,
          Macro<"stderr">,
          Macro<"stdout">,
          Macro<"_IOFBF">,
          Macro<"_IOLBF">,
          Macro<"_IONBF">,
          Macro<"EOF">,
      ], // Macros
      [ // Types
          SizeTType,
          FILE,
      ],
      [], // Enumerations
      [
          FunctionSpec<
              "clearerr",
              RetValSpec<VoidType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "fclose",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "feof",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "ferror",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "fgetc",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "fgets",
              RetValSpec<CharPtr>,
              [
                ArgSpec<CharRestrictedPtr>,
                ArgSpec<IntType>,
                ArgSpec<FILERestrictedPtr>,
              ]
          >,
          FunctionSpec<
              "fflush",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "fopen",
              RetValSpec<FILEPtr>,
              [ArgSpec<ConstCharPtr>,
               ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "fputc",
              RetValSpec<IntType>,
              [ArgSpec<IntType>,
               ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "ftell",
              RetValSpec<LongType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "getc",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "getchar",
              RetValSpec<IntType>,
              [ArgSpec<VoidType>]
          >,
          FunctionSpec<
              "putc",
              RetValSpec<IntType>,
              [ArgSpec<IntType>,
               ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "putchar",
              RetValSpec<IntType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "fputs",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<FILERestrictedPtr>]
          >,
          FunctionSpec<
              "puts",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharRestrictedPtr>]
          >,
          FunctionSpec<
              "fread",
              RetValSpec<SizeTType>,
              [ArgSpec<VoidRestrictedPtr>,
               ArgSpec<SizeTType>,
               ArgSpec<SizeTType>,
               ArgSpec<FILERestrictedPtr>]
          >,
          FunctionSpec<
              "fseek",
              RetValSpec<IntType>,
              [ArgSpec<FILEPtr>,
               ArgSpec<LongType>,
               ArgSpec<IntType>]
          >,
          FunctionSpec<
              "fwrite",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstVoidRestrictedPtr>,
               ArgSpec<SizeTType>,
               ArgSpec<SizeTType>,
               ArgSpec<FILERestrictedPtr>]
          >,
          FunctionSpec<
              "remove",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "rename",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharPtr>, ArgSpec<ConstCharPtr>]
          >,
          FunctionSpec<
              "setbuf",
              RetValSpec<VoidType>,
              [ArgSpec<FILERestrictedPtr>, ArgSpec<CharRestrictedPtr>]
          >,
          FunctionSpec<
              "setvbuf",
              RetValSpec<IntType>,
              [ArgSpec<FILERestrictedPtr>, ArgSpec<CharRestrictedPtr>, ArgSpec<IntType>, ArgSpec<SizeTType>]
          >,
          FunctionSpec<
              "sscanf",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
          FunctionSpec<
              "vsscanf",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VaListType>]
          >,
          FunctionSpec<
              "scanf",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
          FunctionSpec<
              "vscanf",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VaListType>]
          >,
          FunctionSpec<
              "fscanf",
              RetValSpec<IntType>,
              [ArgSpec<FILERestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
          FunctionSpec<
              "vfscanf",
              RetValSpec<IntType>,
              [ArgSpec<FILERestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VaListType>]
          >,
          FunctionSpec<
              "sprintf",
              RetValSpec<IntType>,
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
          FunctionSpec<
              "snprintf",
              RetValSpec<IntType>,
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<SizeTType>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
          FunctionSpec<
              "printf",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
          FunctionSpec<
              "fprintf",
              RetValSpec<IntType>,
              [ArgSpec<FILERestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
          FunctionSpec<
              "asprintf",
              RetValSpec<IntType>,
              [ArgSpec<CharRestrictedPtrPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VarArgType>]
          >,
          FunctionSpec<
              "vsprintf",
              RetValSpec<IntType>,
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VaListType>]
          >,
          FunctionSpec<
              "vsnprintf",
              RetValSpec<IntType>,
              [ArgSpec<CharRestrictedPtr>,
               ArgSpec<SizeTType>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VaListType>]
          >,
          FunctionSpec<
              "vprintf",
              RetValSpec<IntType>,
              [ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VaListType>]
          >,
          FunctionSpec<
              "vfprintf",
              RetValSpec<IntType>,
              [ArgSpec<FILERestrictedPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VaListType>]
          >,
          FunctionSpec<
              "ungetc",
              RetValSpec<IntType>,
              [ArgSpec<IntType>, ArgSpec<FILEPtr>]
          >,
          FunctionSpec<
              "vasprintf",
              RetValSpec<IntType>,
              [ArgSpec<CharRestrictedPtrPtr>,
               ArgSpec<ConstCharRestrictedPtr>,
               ArgSpec<VaListType>]
          >,
      ],
      [
          ObjectSpec<
              "stdin",
              "FILE *"
          >,
          ObjectSpec<
              "stdout",
              "FILE *"
          >,
          ObjectSpec<
              "stderr",
              "FILE *"
          >,
      ]
  >;

  HeaderSpec StdBit = HeaderSpec<
      "stdbit.h",
      [
        Macro<"__STDC_VERSION_STDBIT_H__">,
        Macro<"__STDC_ENDIAN_LITTLE__">,
        Macro<"__STDC_ENDIAN_BIG__">,
        Macro<"__STDC_ENDIAN_NATIVE__">,
        Macro<"stdc_leading_zeros">,
        Macro<"stdc_leading_ones">,
        Macro<"stdc_trailing_zeros">,
        Macro<"stdc_trailing_ones">,
        Macro<"stdc_first_leading_zero">,
        Macro<"stdc_first_leading_one">,
        Macro<"stdc_first_trailing_zero">,
        Macro<"stdc_first_trailing_one">,
        Macro<"stdc_count_zeros">,
        Macro<"stdc_count_ones">,
        Macro<"stdc_has_single_bit">,
        Macro<"stdc_bit_width">,
        Macro<"stdc_bit_floor">,
        Macro<"stdc_bit_ceil">
      ], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<"stdc_leading_zeros_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_leading_zeros_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_leading_zeros_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_leading_zeros_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_leading_zeros_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_leading_ones_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_leading_ones_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_leading_ones_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_leading_ones_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_leading_ones_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_trailing_zeros_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_trailing_zeros_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_trailing_zeros_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_trailing_zeros_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_trailing_zeros_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_trailing_ones_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_trailing_ones_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_trailing_ones_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_trailing_ones_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_trailing_ones_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_first_leading_zero_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_first_leading_zero_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_first_leading_zero_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_first_leading_zero_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_first_leading_zero_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_first_leading_one_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_first_leading_one_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_first_leading_one_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_first_leading_one_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_first_leading_one_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_first_trailing_one_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_first_trailing_one_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_first_trailing_one_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_first_trailing_one_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_first_trailing_one_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_count_zeros_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_count_zeros_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_count_zeros_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_count_zeros_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_count_zeros_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_count_ones_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_count_ones_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_count_ones_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_count_ones_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_count_ones_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_has_single_bit_uc", RetValSpec<BoolType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_has_single_bit_us", RetValSpec<BoolType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_has_single_bit_ui", RetValSpec<BoolType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_has_single_bit_ul", RetValSpec<BoolType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_has_single_bit_ull", RetValSpec<BoolType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_bit_width_uc", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_bit_width_us", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_bit_width_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_bit_width_ul", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_bit_width_ull", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_bit_floor_uc", RetValSpec<UnsignedCharType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_bit_floor_us", RetValSpec<UnsignedShortType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_bit_floor_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_bit_floor_ul", RetValSpec<UnsignedLongType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_bit_floor_ull", RetValSpec<UnsignedLongLongType>, [ArgSpec<UnsignedLongLongType>]>,
          FunctionSpec<"stdc_bit_ceil_uc", RetValSpec<UnsignedCharType>, [ArgSpec<UnsignedCharType>]>,
          FunctionSpec<"stdc_bit_ceil_us", RetValSpec<UnsignedShortType>, [ArgSpec<UnsignedShortType>]>,
          FunctionSpec<"stdc_bit_ceil_ui", RetValSpec<UnsignedIntType>, [ArgSpec<UnsignedIntType>]>,
          FunctionSpec<"stdc_bit_ceil_ul", RetValSpec<UnsignedLongType>, [ArgSpec<UnsignedLongType>]>,
          FunctionSpec<"stdc_bit_ceil_ull", RetValSpec<UnsignedLongLongType>, [ArgSpec<UnsignedLongLongType>]>
      ] // Functions
  >;

  HeaderSpec StdCkdInt = HeaderSpec<
      "stdckdint.h",
      [
        Macro<"__STDC_VERSION_STDCKDINT_H__">,
        Macro<"ckd_add">,
        Macro<"ckd_sub">,
        Macro<"ckd_mul">
      ], // Macros
      [], // Types
      [], // Enumerations
      [] // Functions
  >;

  HeaderSpec StdLib = HeaderSpec<
      "stdlib.h",
      [], // Macros
      [
          DivTType,
          LDivTType,
          LLDivTType,
          SizeTType,
          BSearchCompareT,
          QSortCompareT,
          AtexitHandlerT,
      ], // Types
      [], // Enumerations
      [
          FunctionSpec<"abort", RetValSpec<NoReturn>, [ArgSpec<VoidType>]>,

          FunctionSpec<"bsearch", RetValSpec<VoidPtr>, [ArgSpec<ConstVoidPtr>, ArgSpec<ConstVoidPtr>, ArgSpec<SizeTType>, ArgSpec<SizeTType>, ArgSpec<BSearchCompareT>]>,

          FunctionSpec<"abs", RetValSpec<IntType>, [ArgSpec<IntType>]>,
          FunctionSpec<"labs", RetValSpec<LongType>, [ArgSpec<LongType>]>,
          FunctionSpec<"llabs", RetValSpec<LongLongType>, [ArgSpec<LongLongType>]>,

          FunctionSpec<"atof", RetValSpec<DoubleType>, [ArgSpec<ConstCharRestrictedPtr>]>,
          FunctionSpec<"atoi", RetValSpec<IntType>, [ArgSpec<ConstCharPtr>]>,
          FunctionSpec<"atol", RetValSpec<LongType>, [ArgSpec<ConstCharPtr>]>,
          FunctionSpec<"atoll", RetValSpec<LongLongType>, [ArgSpec<ConstCharPtr>]>,

          FunctionSpec<"div", RetValSpec<DivTType>, [ArgSpec<IntType>, ArgSpec<IntType>]>,
          FunctionSpec<"ldiv", RetValSpec<LDivTType>, [ArgSpec<LongType>, ArgSpec<LongType>]>,
          FunctionSpec<"lldiv", RetValSpec<LLDivTType>, [ArgSpec<LongLongType>, ArgSpec<LongLongType>]>,

          FunctionSpec<"qsort", RetValSpec<VoidType>, [ArgSpec<VoidPtr>, ArgSpec<SizeTType>, ArgSpec<SizeTType>, ArgSpec<QSortCompareT>]>,

          FunctionSpec<"rand", RetValSpec<IntType>, [ArgSpec<VoidType>]>,
          FunctionSpec<"srand", RetValSpec<VoidType>, [ArgSpec<UnsignedIntType>]>,

          FunctionSpec<"strfromf", RetValSpec<IntType>, [ArgSpec<CharRestrictedPtr>, ArgSpec<SizeTType>, ArgSpec<ConstCharRestrictedPtr>, ArgSpec<FloatType>]>,
          FunctionSpec<"strfromd", RetValSpec<IntType>, [ArgSpec<CharRestrictedPtr>, ArgSpec<SizeTType>, ArgSpec<ConstCharRestrictedPtr>, ArgSpec<DoubleType>]>,
          FunctionSpec<"strfroml", RetValSpec<IntType>, [ArgSpec<CharRestrictedPtr>, ArgSpec<SizeTType>, ArgSpec<ConstCharRestrictedPtr>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"strtof", RetValSpec<FloatType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>]>,
          FunctionSpec<"strtod", RetValSpec<DoubleType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>]>,
          FunctionSpec<"strtold", RetValSpec<LongDoubleType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>]>,
          FunctionSpec<"strtol", RetValSpec<LongType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>]>,
          FunctionSpec<"strtoll", RetValSpec<LongLongType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>]>,
          FunctionSpec<"strtoul", RetValSpec<UnsignedLongType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>]>,
          FunctionSpec<"strtoull", RetValSpec<UnsignedLongLongType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>]>,

          FunctionSpec<"strtof_l", RetValSpec<FloatType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<LocaleT>]>,
          FunctionSpec<"strtod_l", RetValSpec<DoubleType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<LocaleT>]>,
          FunctionSpec<"strtold_l", RetValSpec<LongDoubleType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<LocaleT>]>,
          FunctionSpec<"strtol_l", RetValSpec<LongType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>, ArgSpec<LocaleT>]>,
          FunctionSpec<"strtoll_l", RetValSpec<LongLongType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>, ArgSpec<LocaleT>]>,
          FunctionSpec<"strtoul_l", RetValSpec<UnsignedLongType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>, ArgSpec<LocaleT>]>,
          FunctionSpec<"strtoull_l", RetValSpec<UnsignedLongLongType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>, ArgSpec<LocaleT>]>,

          FunctionSpec<"malloc", RetValSpec<VoidPtr>, [ArgSpec<SizeTType>]>,
          FunctionSpec<"calloc", RetValSpec<VoidPtr>, [ArgSpec<SizeTType>, ArgSpec<SizeTType>]>,
          FunctionSpec<"realloc", RetValSpec<VoidPtr>, [ArgSpec<VoidPtr>, ArgSpec<SizeTType>]>,
          FunctionSpec<"aligned_alloc", RetValSpec<VoidPtr>, [ArgSpec<SizeTType>, ArgSpec<SizeTType>]>,
          FunctionSpec<"free", RetValSpec<VoidType>, [ArgSpec<VoidPtr>]>,

          FunctionSpec<"_Exit", RetValSpec<NoReturn>, [ArgSpec<IntType>]>,
          FunctionSpec<"at_quick_exit", RetValSpec<IntType>, [ArgSpec<AtexitHandlerT>]>,
          FunctionSpec<"atexit", RetValSpec<IntType>, [ArgSpec<AtexitHandlerT>]>,
          FunctionSpec<"exit", RetValSpec<NoReturn>, [ArgSpec<IntType>]>,
          FunctionSpec<"quick_exit", RetValSpec<NoReturn>, [ArgSpec<IntType>]>,

          FunctionSpec<"system", RetValSpec<IntType>, [ArgSpec<ConstCharPtr>]>,
      ]
  >;

  NamedType IMaxDivTType = NamedType<"imaxdiv_t">;

  HeaderSpec IntTypes = HeaderSpec<
      "inttypes.h",
      [
        Macro<"__STDC_VERSION_INTTYPES_H__">,
      ], // Macros
      [
        IMaxDivTType,
      ], // Types
      [], // Enumerations
      [
          FunctionSpec<"imaxabs", RetValSpec<IntMaxTType>, [ArgSpec<IntMaxTType>]>,
          FunctionSpec<"imaxdiv", RetValSpec<IMaxDivTType>, [ArgSpec<IntMaxTType>, ArgSpec<IntMaxTType>]>,
          FunctionSpec<"strtoimax", RetValSpec<IntMaxTType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>]>,
          FunctionSpec<"strtoumax", RetValSpec<UIntMaxTType>, [ArgSpec<ConstCharRestrictedPtr>, ArgSpec<CharRestrictedPtrPtr>, ArgSpec<IntType>]>,
      ]
  >;

  HeaderSpec Errno = HeaderSpec<
      "errno.h",
      [
        Macro<"errno">,
        Macro<"EDOM">,
        Macro<"EILSEQ">,
        Macro<"ERANGE">,
      ]
  >;

  HeaderSpec Float = HeaderSpec<
      "float.h",
      [
        Macro<"FLT_MANT_DIG">,
        Macro<"DBL_MANT_DIG">,
        Macro<"LDBL_MANT_DIG">,
      ]
  >;

  HeaderSpec StdInt = HeaderSpec<"StdInt.h">;

  HeaderSpec Limits = HeaderSpec<"limits.h">;

  NamedType SigAtomicT = NamedType<"sig_atomic_t">;
  HeaderSpec Signal = HeaderSpec<
      "signal.h",
      [
        Macro<"SIG_BLOCK">,
        Macro<"SIG_UNBLOCK">,
        Macro<"SIG_SETMASK">,

        Macro<"SIGABRT">,
        Macro<"SIGFPE">,
        Macro<"SIGILL">,
        Macro<"SIGINT">,
        Macro<"SIGSEGV">,
        Macro<"SIGTERM">
      ],
      [
        SizeTType,
        SigAtomicT,
        SigHandlerT,
      ],
      [], // Enumerations
      [
        FunctionSpec<"raise", RetValSpec<IntType>, [ArgSpec<IntType>]>,
        FunctionSpec<
          "signal",
          RetValSpec<SigHandlerT>,
          [ArgSpec<IntType>, ArgSpec<SigHandlerT>]
        >,
      ]
  >;

  HeaderSpec Threads = HeaderSpec<
      "threads.h",
      [
          Macro<"ONCE_FLAG_INIT">,
      ],
      [
          OnceFlagType,
          CallOnceFuncType,
          CndTType,
          MtxTType,
          ThrdStartTType,
          ThrdTType,
          TssTType,
          TssDtorTType,
      ],
      [
          EnumeratedNameValue<"mtx_plain">,
          EnumeratedNameValue<"mtx_recursive">,
          EnumeratedNameValue<"mtx_timed">,
          EnumeratedNameValue<"thrd_timedout">,
          EnumeratedNameValue<"thrd_success">,
          EnumeratedNameValue<"thrd_busy">,
          EnumeratedNameValue<"thrd_error">,
          EnumeratedNameValue<"thrd_nomem">,
      ],
      [
          FunctionSpec<
              "call_once",
              RetValSpec<VoidType>,
              [
                  ArgSpec<OnceFlagTypePtr>,
                  ArgSpec<CallOnceFuncType>,
              ]
          >,
          FunctionSpec<
              "cnd_broadcast",
              RetValSpec<IntType>,
              [
                  ArgSpec<CndTTypePtr>,
              ]
          >,
          FunctionSpec<
              "cnd_destroy",
              RetValSpec<VoidType>,
              [
                  ArgSpec<CndTTypePtr>,
              ]
          >,
          FunctionSpec<
              "cnd_init",
              RetValSpec<IntType>,
              [
                  ArgSpec<CndTTypePtr>,
              ]
          >,
          FunctionSpec<
              "cnd_signal",
              RetValSpec<IntType>,
              [
                  ArgSpec<CndTTypePtr>,
              ]
          >,
          FunctionSpec<
              "cnd_wait",
              RetValSpec<IntType>,
              [
                  ArgSpec<CndTTypePtr>,
                  ArgSpec<MtxTTypePtr>,
              ]
          >,
          FunctionSpec<
              "mtx_init",
              RetValSpec<IntType>,
              [
                  ArgSpec<MtxTTypePtr>,
                  ArgSpec<IntType>,
              ]
          >,
          FunctionSpec<
              "mtx_destroy",
              RetValSpec<IntType>,
              [
                  ArgSpec<VoidType>,
              ]
          >,
          FunctionSpec<
              "mtx_lock",
              RetValSpec<IntType>,
              [
                  ArgSpec<MtxTTypePtr>,
              ]
          >,
          FunctionSpec<
              "mtx_unlock",
              RetValSpec<IntType>,
              [
                  ArgSpec<MtxTTypePtr>,
              ]
          >,
          FunctionSpec<
              "thrd_create",
              RetValSpec<IntType>,
              [
                  ArgSpec<ThrdTTypePtr>,
                  ArgSpec<ThrdStartTType>,
                  ArgSpec<VoidPtr>,
              ]
          >,
          FunctionSpec<
              "thrd_join",
              RetValSpec<IntType>,
              [
                  ArgSpec<ThrdTType>,
                  ArgSpec<IntPtr>,
              ]
          >,
          FunctionSpec<
              "thrd_detach",
              RetValSpec<IntType>,
              [ArgSpec<ThrdTType>]
          >,
          FunctionSpec<
              "thrd_current",
              RetValSpec<ThrdTType>,
              [ArgSpec<VoidType>]
          >,
          FunctionSpec<
              "thrd_equal",
              RetValSpec<IntType>,
              [ArgSpec<ThrdTType>, ArgSpec<ThrdTType>]
          >,
          FunctionSpec<
              "thrd_exit",
              RetValSpec<VoidType>,
              [ArgSpec<IntType>]
          >,
          FunctionSpec<
              "tss_create",
              RetValSpec<IntType>,
              [ArgSpec<TssTPtr>, ArgSpec<TssDtorTType>]
          >,
          FunctionSpec<
              "tss_delete",
              RetValSpec<IntType>,
              [ArgSpec<TssTType>]
          >,
          FunctionSpec<
              "tss_get",
              RetValSpec<VoidPtr>,
              [ArgSpec<TssTType>]
          >,
          FunctionSpec<
              "tss_set",
              RetValSpec<IntType>,
              [ArgSpec<TssTType>, ArgSpec<VoidPtr>]
          >,
      ]
  >;

  HeaderSpec Time = HeaderSpec<
      "time.h",
      [], // Macros
      [ // Types
         ClockT,
         StructTmType,
         StructTimeSpec,
         TimeTType,
         SizeTType,
      ],
      [], // Enumerations
      [
          FunctionSpec<
              "asctime",
              RetValSpec<CharPtr>,
              [ArgSpec<StructTmPtr>]
          >,
          FunctionSpec<
              "asctime_r",
              RetValSpec<CharPtr>,
              [
                  ArgSpec<StructTmPtr>,
                  ArgSpec<CharPtr>,
              ]
          >,
          FunctionSpec<
              "ctime",
              RetValSpec<CharPtr>,
              [ArgSpec<TimeTTypePtr>]
          >,
          FunctionSpec<
              "ctime_r",
              RetValSpec<CharPtr>,
              [
                  ArgSpec<TimeTTypePtr>,
                  ArgSpec<CharPtr>,
              ]
          >,
          FunctionSpec<
              "localtime",
              RetValSpec<StructTmPtr>,
              [ArgSpec<TimeTTypePtr>]
          >,
          FunctionSpec<
              "localtime_r",
              RetValSpec<StructTmPtr>,
              [
                  ArgSpec<TimeTTypePtr>,
                  ArgSpec<StructTmPtr>,
              ]
          >,
          FunctionSpec<
              "localtime_s",
              RetValSpec<IntType>,
              [
                  ArgSpec<TimeTTypePtr>,
                  ArgSpec<StructTmPtr>,
              ]
          >,
          FunctionSpec<
              "clock",
              RetValSpec<ClockT>,
              [ArgSpec<VoidType>]
          >,
          FunctionSpec<
              "difftime",
              RetValSpec<DoubleType>,
              [
                  ArgSpec<TimeTType>,
                  ArgSpec<TimeTType>,
              ]
          >,
          FunctionSpec<
              "gmtime",
              RetValSpec<StructTmPtr>,
              [ArgSpec<TimeTTypePtr>]
          >,
          FunctionSpec<
              "gmtime_r",
              RetValSpec<StructTmPtr>,
              [
                  ArgSpec<TimeTTypePtr>,
                  ArgSpec<StructTmPtr>,
              ]
          >,
          FunctionSpec<
              "mktime",
              RetValSpec<TimeTType>,
              [ArgSpec<StructTmPtr>]
          >,
          FunctionSpec<
              "time",
              RetValSpec<TimeTType>,
              [ArgSpec<TimeTTypePtr>]
          >,
      ]
  >;

  HeaderSpec SetJmp = HeaderSpec<
      "setjmp.h",
      [], // Macros
      [JmpBuf],
      [], // Enumerations
      [
          FunctionSpec<
              "longjmp",
              RetValSpec<NoReturn>,
              [ArgSpec<JmpBuf>, ArgSpec<IntType>]
          >,
          FunctionSpec<
              "setjmp",
              RetValSpec<IntType>,
              [ArgSpec<JmpBuf>]
          >,
          FunctionSpec<
              "longjmp",
              RetValSpec<VoidType>,
              [ArgSpec<JmpBuf>, ArgSpec<IntType>]
          >,
      ]
  >;

  HeaderSpec UChar = HeaderSpec<
      "uchar.h",
      [], // Macros
      [ //Types
        MBStateTType,
        Char8TType,
        Char16TType,
        Char32TType,
        SizeTType,
      ],
      [], // Enumerations
      []
  >;

  HeaderSpec WChar = HeaderSpec<
      "wchar.h",
      [ // Macros
        Macro<"WEOF">,
      ],
      [ //Types
        MBStateTType,
        SizeTType,
        WIntType,
        WCharType,
      ],
      [], // Enumerations
      [
          FunctionSpec<
              "wctob",
              RetValSpec<IntType>,
              [ArgSpec<WIntType>]
          >,
      ]
  >;

  
  NamedType StructLconv = NamedType<"struct lconv">;
  PtrType StructLconvPtr = PtrType<StructLconv>;

  HeaderSpec Locale = HeaderSpec<
     "locale.h",
      [], // Macros
      [LocaleT, StructLconv], // Types
      [], // Enumerations
      [
        FunctionSpec<
          "duplocale",
          RetValSpec<LocaleT>,
          [
            ArgSpec<LocaleT>
          ]
        >,
        FunctionSpec<
          "freelocale",
          RetValSpec<VoidType>,
          [
            ArgSpec<LocaleT>
          ]
        >,
        FunctionSpec<
          "localeconv",
          RetValSpec<StructLconvPtr>,
          []
        >,
        FunctionSpec<
          "newlocale",
          RetValSpec<LocaleT>,
          [
            ArgSpec<IntType>,
            ArgSpec<ConstCharPtr>,
            ArgSpec<LocaleT>
          ]
        >,
        FunctionSpec<
          "setlocale",
          RetValSpec<CharPtr>,
          [
            ArgSpec<IntType>,
            ArgSpec<ConstCharPtr>
          ]
        >,
        FunctionSpec<
          "uselocale",
          RetValSpec<LocaleT>,
          [
            ArgSpec<LocaleT>
          ]
        >
      ]  // Functions
  >;

  let Headers = [
    Assert,
    CType,
    Errno,
    Fenv,
    Float,
    StdInt,
    Limits,
    Math,
    String,
    StdBit,
    StdCkdInt,
    StdIO,
    StdLib,
    IntTypes,
    SetJmp,
    Signal,
    Threads,
    Time,
    UChar,
    WChar,
    Locale,
  ];
}

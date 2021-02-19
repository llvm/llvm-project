def StdC : StandardSpec<"stdc"> {

  NamedType FILE = NamedType<"FILE">;
  PtrType FILEPtr = PtrType<FILE>;
  RestrictedPtrType FILERestrictedPtr = RestrictedPtrType<FILE>;
  NamedType StructTmType = NamedType<"struct tm">;
  PtrType StructTmPtr = PtrType<StructTmType>;

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
  
  HeaderSpec CType = HeaderSpec<
      "ctype.h",
      [], // Macros
      [], // Types
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
          Macro<"FE_UPWARD">
      ],
      [
          NamedType<"fenv_t">,
          NamedType<"fexcept_t">,
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

          Macro<"INFINITY">,
          Macro<"NAN">,

          Macro<"FP_ILOGB0">,
          Macro<"FP_ILOGBNAN">,

          Macro<"isfinite">,
          Macro<"isinf">,
          Macro<"isnan">,
      ],
      [
          NamedType<"float_t">,
          NamedType<"double_t">,
      ],
      [], // Enumerations
      [
          FunctionSpec<"copysign", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"copysignf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"copysignl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"ceil", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"ceilf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"ceill", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"fabs", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"fabsf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"fabsl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"fdim", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fdimf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fdiml", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"floor", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"floorf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"floorl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"fmin", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fminf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fminl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"fmax", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"fmaxf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"fmaxl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"fmaf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>, ArgSpec<FloatType>]>,

          FunctionSpec<"frexp", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntPtr>]>,
          FunctionSpec<"frexpf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntPtr>]>,
          FunctionSpec<"frexpl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntPtr>]>,

          FunctionSpec<"hypot", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"hypotf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,

          FunctionSpec<"ilogb", RetValSpec<IntType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"ilogbf", RetValSpec<IntType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"ilogbl", RetValSpec<IntType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"ldexp", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<IntType>]>,
          FunctionSpec<"ldexpf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<IntType>]>,
          FunctionSpec<"ldexpl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<IntType>]>,

          FunctionSpec<"logb", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"logbf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"logbl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"modf", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoublePtr>]>,
          FunctionSpec<"modff", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatPtr>]>,
          FunctionSpec<"modfl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoublePtr>]>,

          FunctionSpec<"cosf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"sinf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"expf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"exp2f", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,

          FunctionSpec<"remainderf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"remainder", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"remainderl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,

          FunctionSpec<"remquof", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>, ArgSpec<IntPtr>]>,
          FunctionSpec<"remquo", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>, ArgSpec<IntPtr>]>,
          FunctionSpec<"remquol", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>, ArgSpec<IntPtr>]>,

          FunctionSpec<"round", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"roundf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"roundl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"lround", RetValSpec<LongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"lroundf", RetValSpec<LongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"lroundl", RetValSpec<LongType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"llround", RetValSpec<LongLongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"llroundf", RetValSpec<LongLongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"llroundl", RetValSpec<LongLongType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"rint", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"rintf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"rintl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"lrint", RetValSpec<LongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"lrintf", RetValSpec<LongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"lrintl", RetValSpec<LongType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"llrint", RetValSpec<LongLongType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"llrintf", RetValSpec<LongLongType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"llrintl", RetValSpec<LongLongType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"sqrt", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"sqrtf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"sqrtl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"trunc", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"truncf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"truncl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"nearbyint", RetValSpec<DoubleType>, [ArgSpec<DoubleType>]>,
          FunctionSpec<"nearbyintf", RetValSpec<FloatType>, [ArgSpec<FloatType>]>,
          FunctionSpec<"nearbyintl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>]>,

          FunctionSpec<"nextafterf", RetValSpec<FloatType>, [ArgSpec<FloatType>, ArgSpec<FloatType>]>,
          FunctionSpec<"nextafter", RetValSpec<DoubleType>, [ArgSpec<DoubleType>, ArgSpec<DoubleType>]>,
          FunctionSpec<"nextafterl", RetValSpec<LongDoubleType>, [ArgSpec<LongDoubleType>, ArgSpec<LongDoubleType>]>,
      ]
  >;

  HeaderSpec StdIO = HeaderSpec<
      "stdio.h",
      [], // Macros
      [ // Types
          SizeTType,
          FILE,
      ],
      [], // Enumerations
      [
          FunctionSpec<
              "fwrite",
              RetValSpec<SizeTType>,
              [ArgSpec<ConstVoidRestrictedPtr>,
              ArgSpec<SizeTType>,
              ArgSpec<SizeTType>,
              ArgSpec<FILERestrictedPtr>]
          >,
      ]
  >;

  HeaderSpec StdLib = HeaderSpec<
      "stdlib.h",
      [], // Macros
      [], // Types
      [], // Enumerations
      [
          FunctionSpec<"abort", RetValSpec<NoReturn>, [ArgSpec<VoidType>]>,
          FunctionSpec<"abs", RetValSpec<IntType>, [ArgSpec<IntType>]>,
          FunctionSpec<"labs", RetValSpec<LongType>, [ArgSpec<LongType>]>,
          FunctionSpec<"llabs", RetValSpec<LongLongType>, [ArgSpec<LongLongType>]>,
          FunctionSpec<"_Exit", RetValSpec<NoReturn>, [ArgSpec<IntType>]>,
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
          MtxTType,
          ThrdStartTType,
          ThrdTType,
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
              "mtx_init",
              RetValSpec<IntType>,
              [
                  ArgSpec<MtxTTypePtr>,
                  ArgSpec<IntType>,
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
                  ArgSpec<ThrdTTypePtr>,
                  ArgSpec<IntPtr>,
              ]
          >
      ]
  >;

  HeaderSpec Time = HeaderSpec<
      "time.h",
      [], // Macros
      [ // Types
       StructTmType,
       TimeTType,
      ],
      [], // Enumerations
      [
          FunctionSpec<
              "mktime",
              RetValSpec<TimeTType>,
              [ArgSpec<StructTmPtr>]
          >,
      ]
  >;

  let Headers = [
    Assert,
    CType,
    Errno,
    Fenv,
    Math,
    String,
    StdIO,
    StdLib,
    Signal,
    Threads,
    Time,
  ];
}

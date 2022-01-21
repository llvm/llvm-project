include "config/public_api.td"

include "spec/gnu_ext.td"
include "spec/linux.td"
include "spec/llvm_libc_ext.td"
include "spec/posix.td"
include "spec/stdc.td"

// TODO: Eliminate all TypeDecl specializations. Since we define all public
// types in their own self contained header files, the header generator can
// produce the boiler plate which pulls in the type definitions.

def SizeT : TypeDecl<"size_t"> {
  let Decl = [{
    #include <llvm-libc-types/size_t.h>
  }];
}

def SSizeT : TypeDecl<"ssize_t"> {
  let Decl = [{
    #include <llvm-libc-types/ssize_t.h>
  }];
}

def StructTm: TypeDecl<"struct tm"> {
  let Decl = [{
    #include <llvm-libc-types/tm.h>
  }];
}

def TimeT: TypeDecl<"time_t"> {
  let Decl = [{
    #include <llvm-libc-types/time_t.h>
  }];
}

def OffT : TypeDecl<"off_t"> {
  let Decl = [{
    #include <llvm-libc-types/off_t.h>
  }];
}

def FILE : TypeDecl<"FILE"> {
  let Decl = [{
    #include <llvm-libc-types/FILE.h>
  }];
}

def AssertMacro : MacroDef<"assert"> {
  let Defn = [{
    #undef assert

    #ifdef NDEBUG
    #define assert(e) (void)0
    #else

    #ifdef __cplusplus
    extern "C"
    #endif
    _Noreturn void __assert_fail(const char *, const char *, unsigned, const char *);

    #define assert(e)  \
      ((e) ? (void)0 : __assert_fail(#e, __FILE__, __LINE__, __PRETTY_FUNCTION__))

    #endif
  }];
}

def StaticAssertMacro : MacroDef<"static_assert"> {
  let Defn = [{
    #ifndef __cplusplus
    #undef static_assert
    #define static_assert _Static_assert
    #endif
  }];
}

def NullMacro : MacroDef<"NULL"> {
  let Defn = [{
    #define __need_NULL
    #include <stddef.h>
  }];
}

def ErrnoMacro : MacroDef<"errno"> {
  let Defn = [{
    #ifdef __cplusplus
    extern "C"
    #endif
    int *__errno_location();
    #define errno (*__errno_location())
  }];
}

def AssertAPI : PublicAPI<"assert.h"> {
  let Macros = [
    AssertMacro,
    StaticAssertMacro,
  ];
}

def CTypeAPI : PublicAPI<"ctype.h"> {
}

def IMaxDivT : TypeDecl<"imaxdiv_t"> {
  let Decl = [{
    #include <llvm-libc-types/imaxdiv_t.h>
  }];
}

def IntTypesAPI : PublicAPI<"inttypes.h"> {
  let TypeDeclarations = [
    IMaxDivT,
  ];
}

def MathErrHandlingMacro : MacroDef<"math_errhandling"> {
  let Defn = [{
    #ifndef math_errhandling
    #ifdef __FAST_MATH__
    #define math_errhandling 0
    #elif defined __NO_MATH_ERRNO__
    #define math_errhandling (MATH_ERREXCEPT)
    #else
    #define math_errhandling (MATH_ERRNO | MATH_ERREXCEPT)
    #endif
    #endif // math_errhandling not defined
  }];
}

def IsFiniteMacro : MacroDef<"isfinite"> {
  let Defn = [{
    #define isfinite(x) __builtin_isfinite(x)
  }];
}

def IsInfMacro : MacroDef<"isinf"> {
  let Defn = [{
    #define isinf(x) __builtin_isinf(x)
  }];
}

def IsNanMacro : MacroDef<"isnan"> {
  let Defn = [{
    #define isnan(x) __builtin_isnan(x)
  }];
}

def FloatT : TypeDecl<"float_t"> {
  let Decl = [{
    #include <llvm-libc-types/float_t.h>
  }];
}

def DoubleT : TypeDecl<"double_t"> {
  let Decl = [{
    #include <llvm-libc-types/double_t.h>
  }];
}

def MathAPI : PublicAPI<"math.h"> {
  let Macros = [
    SimpleMacroDef<"MATH_ERRNO", "1">,
    SimpleMacroDef<"MATH_ERREXCEPT", "2">,
    MathErrHandlingMacro,

    SimpleMacroDef<"INFINITY", "__builtin_inff()">,
    SimpleMacroDef<"NAN", "__builtin_nanf(\"\")">,

    SimpleMacroDef<"FP_ILOGB0", "(-__INT_MAX__ - 1)">, // INT_MIN
    SimpleMacroDef<"FP_ILOGBNAN", "__INT_MAX__">,

    IsFiniteMacro,
    IsInfMacro,
    IsNanMacro,
  ];
  let TypeDeclarations = [
    DoubleT,
    FloatT,
  ];
}

def FEnvT : TypeDecl<"fenv_t"> {
  let Decl = [{
    #include <llvm-libc-types/fenv_t.h>
  }];
}

def FExceptT : TypeDecl<"fexcept_t"> {
  let Decl = [{
    #include <llvm-libc-types/fexcept_t.h>
  }];
}

def FenvAPI: PublicAPI<"fenv.h"> {
  let Macros = [
    SimpleMacroDef<"FE_DIVBYZERO", "1">,
    SimpleMacroDef<"FE_INEXACT", "2">,
    SimpleMacroDef<"FE_INVALID", "4">,
    SimpleMacroDef<"FE_OVERFLOW", "8">,
    SimpleMacroDef<"FE_UNDERFLOW", "16">,
    SimpleMacroDef<"FE_ALL_EXCEPT", "(FE_DIVBYZERO|FE_INEXACT|FE_INVALID|FE_OVERFLOW|FE_UNDERFLOW)">,

    SimpleMacroDef<"FE_DOWNWARD", "1">,
    SimpleMacroDef<"FE_TONEAREST", "2">,
    SimpleMacroDef<"FE_TOWARDZERO", "4">,
    SimpleMacroDef<"FE_UPWARD", "8">,

    SimpleMacroDef<"FE_DFL_ENV", "((fenv_t *)-1)">,
  ];
  let TypeDeclarations = [
    FEnvT,
    FExceptT,
  ];
}

def StringAPI : PublicAPI<"string.h"> {
  let TypeDeclarations = [
    SizeT,
  ];

  let Macros = [
    NullMacro,
  ];
}

def StdIOAPI : PublicAPI<"stdio.h"> {
  let TypeDeclarations = [
    SizeT,
    FILE,
  ];
}

def DivT : TypeDecl<"div_t"> {
  let Decl = [{
    #include <llvm-libc-types/div_t.h>
  }];
}

def LDivT : TypeDecl<"ldiv_t"> {
  let Decl = [{
    #include <llvm-libc-types/ldiv_t.h>
  }];
}

def LLDivT : TypeDecl<"lldiv_t"> {
  let Decl = [{
    #include <llvm-libc-types/lldiv_t.h>
  }];
}

def BSearchCompareTDefn : TypeDecl<"__bsearchcompare_t"> {
  let Decl = [{
    #include <llvm-libc-types/__bsearchcompare_t.h>
  }];
}

def QSortCompareTDefn : TypeDecl<"__qsortcompare_t"> {
  let Decl = [{
    #include <llvm-libc-types/__qsortcompare_t.h>
  }];
}

def StdlibAPI : PublicAPI<"stdlib.h"> {
  let TypeDeclarations = [
    DivT,
    LDivT,
    LLDivT,
    SizeT,
    BSearchCompareTDefn,
    QSortCompareTDefn,
  ];
}

def TimeAPI : PublicAPI<"time.h"> {
  let TypeDeclarations = [
    StructTm,
    TimeT,
  ];

  let Functions = [
    "asctime",
    "asctime_r",
    "gmtime",
    "gmtime_r",
    "mktime",
  ];
}

def ErrnoAPI : PublicAPI<"errno.h"> {
  let Macros = [
    ErrnoMacro,
    // We largely depend on linux/errno.h to give us the
    // various error macro definitions. However, some libc
    // implementations have chosen to provide definitions
    // for some of the error macros to account for the ones
    // missing in linux/errno.h. There is no harm in doing
    // the same here if we define the macros only when they
    // are not already defined.
    MacroDefineIfNot<"ENOTSUP", "EOPNOTSUPP">,
    MacroDefineIfNot<"ECANCELED", "125">,
    MacroDefineIfNot<"EOWNERDEAD", "130">,
    MacroDefineIfNot<"ENOTRECOVERABLE", "131">,
    MacroDefineIfNot<"ERFKILL", "132">,
    MacroDefineIfNot<"EHWPOISON", "133">,
  ];
}

def SysMManAPI : PublicAPI<"sys/mman.h"> {
  let Macros = [
    SimpleMacroDef<"PROT_NONE", "0">,
    SimpleMacroDef<"PROT_READ", "1">,
    SimpleMacroDef<"PROT_WRITE", "2">,
    SimpleMacroDef<"PROT_EXEC", "4">,

    SimpleMacroDef<"MAP_FIXED", "1">,
    SimpleMacroDef<"MAP_PRIVATE", "2">,
    SimpleMacroDef<"MAP_SHARED", "4">,

    SimpleMacroDef<"MAP_FAILED", "((void*)-1)">,

    // TODO: The value of 0x20 is good for x86_64, but has to be extended
    // in some manner to accommodate other machine architectures.
    SimpleMacroDef<"MAP_ANONYMOUS", "0x20">

    // TODO: Add other MAP_* macros used by Linux.
  ];

  let TypeDeclarations = [
    SizeT,
    OffT,
  ];
}

def StructSigactionDefn : TypeDecl<"struct sigaction"> {
  let Decl = [{
    #include <llvm-libc-types/__sigaction.h>
  }];
}

def SighandlerTDefn : TypeDecl<"__sighandler_t"> {
  let Decl = [{
    #include <llvm-libc-types/__sighandler_t.h>
  }];
}

def SignalAPI : PublicAPI<"signal.h"> {
  let TypeDeclarations = [
    StructSigactionDefn,
    SighandlerTDefn,
  ];
}

def OnceFlag : TypeDecl<"once_flag"> {
  let Decl = [{
    #include <llvm-libc-types/once_flag.h>
  }];
}

def MtxT : TypeDecl<"mtx_t"> {
  let Decl = [{
    #include <llvm-libc-types/mtx_t.h>
  }];
}

def CndT : TypeDecl<"cnd_t"> {
  let Decl = [{
    #include <llvm-libc-types/cnd_t.h>
  }];
}

def ThrdT : TypeDecl<"thrd_t"> {
  let Decl = [{
    #include <llvm-libc-types/thrd_t.h>
  }];
}

def ThreadStartT : TypeDecl<"thrd_start_t"> {
  let Decl = [{
    #include <llvm-libc-types/thrd_start_t.h>
  }];
}

def CallOnceFuncT : TypeDecl<"__call_once_func_t"> {
  let Decl = [{
    #include <llvm-libc-types/__call_once_func_t.h>
  }];
}

def ThreadsAPI : PublicAPI<"threads.h"> {
  let Macros = [
    SimpleMacroDef<"ONCE_FLAG_INIT", "0">,
  ];

  let TypeDeclarations = [
    OnceFlag,
    CallOnceFuncT,
    MtxT,
    CndT,
    ThrdT,
    ThreadStartT,
  ];

  let Enumerations = [
    "mtx_plain",
    "mtx_recursive",
    "mtx_timed",
    "thrd_timedout",
    "thrd_success",
    "thrd_busy",
    "thrd_error",
    "thrd_nomem",
  ];
}

def UniStdAPI : PublicAPI<"unistd.h"> {
  let TypeDeclarations = [
    SSizeT,
    SizeT,
  ];
}

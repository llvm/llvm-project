include "config/public_api.td"

include "spec/stdc.td"

def AssertMacro : MacroDef<"assert"> {
  let Defn = [{
    #undef assert

    #ifdef NDEBUG
    #define assert(e) (void)0
    #else

    #ifdef __cplusplus
    extern "C"
    #endif
    _Noreturn void __assert_fail(const char *, const char *, unsigned, const char *) __NOEXCEPT;

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

def AssertAPI : PublicAPI<"assert.h"> {
  let Macros = [
    AssertMacro,
    StaticAssertMacro,
  ];
}

def CTypeAPI : PublicAPI<"ctype.h"> {
}

def FEnvAPI : PublicAPI<"fenv.h"> {
  let Types = ["fenv_t", "fexcept_t"];
}

def IntTypesAPI : PublicAPI<"inttypes.h"> {
  let Types = ["imaxdiv_t"];
}

def StdlibAPI : PublicAPI<"stdlib.h"> {
  let Types = [
    "div_t",
    "ldiv_t",
    "lldiv_t",
    "size_t",
    "__bsearchcompare_t",
    "__qsortcompare_t",
    "__atexithandler_t",
  ];
}

def StringAPI : PublicAPI<"string.h"> {
  let Types = ["size_t"];
}

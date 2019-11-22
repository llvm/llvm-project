include "config/public_api.td"

include "spec/stdc.td"

def FloatT : TypeDecl<"float_t"> {
  let Decl = [{
    #if __FLT_EVAL_METHOD__ == 1
      typedef float float_t
    #elif __FLT_EVAL_METHOD__ == 2
      ...
    #else
      ...
    #endif
  }]; // This is only an example and not exactly how it will appear
}

def SizeT : TypeDecl<"size_t"> {
  let Decl = [{
    #define __need_size_t
    #include <stddef.h>
  }];
}

def NullMacro : MacroDef<"NULL"> {
  let Defn = [{
    #define __need_NULL
    #include <stddef.h>
  }];
}

def MathAPI : PublicAPI<"math.h"> {
  let Functions = [
   "acos",
   "acosl",
  ];

  let TypeDeclarations = [
    FloatT,
  ];
}

def StringAPI : PublicAPI<"string.h"> {
  let Functions = [
    "memcpy",
    "memmove",
    "memcmp",
    "memchr",
    "memset",
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "strcmp",
    "strcoll",
    "strncmp",
    "strxfrm",
    "strchr",
    "strcspn",
    "strpbrk",
    "strrchr",
    "strspn",
    "strstr",
    "strtok",
    "strerror",
    "strlen",
  ];

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
  ];

  let Functions = [
    "snprintf",
  ];
}

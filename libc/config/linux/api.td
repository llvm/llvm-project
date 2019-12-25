include "config/public_api.td"

include "spec/linux.td"
include "spec/posix.td"
include "spec/stdc.td"

def SizeT : TypeDecl<"size_t"> {
  let Decl = [{
    #define __need_size_t
    #include <stddef.h>
  }];
}

def OffT : TypeDecl<"off_t"> {
  let Decl = [{
    #define __need_off_t
    #include <__posix-types.h>
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
    int *__errno_location();
    #define errno (*__errno_location())
  }];
}

def MathAPI : PublicAPI<"math.h"> {
  let Functions = [
   "acos",
   "acosl",
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

  let Functions = [
    "mmap",
    "munmap",
  ];
}

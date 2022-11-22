include "config/public_api.td"

include "spec/stdc.td"

def NullMacro : MacroDef<"NULL"> {
  let Defn = [{
    #define __need_NULL
    #include <stddef.h>
  }];
}

def StringAPI : PublicAPI<"string.h"> {
  let Types = ["size_t"];

  let Macros = [
    NullMacro,
  ];
}

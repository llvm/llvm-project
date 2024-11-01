include "config/public_api.td"

include "spec/stdc.td"

def StringAPI : PublicAPI<"string.h"> {
  let Types = ["size_t"];
}

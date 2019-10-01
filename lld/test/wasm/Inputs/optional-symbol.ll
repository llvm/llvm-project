target triple = "wasm32-unknown-unknown"

@__dso_handle = external global i8*

define i8** @get_optional() {
  ret i8** @__dso_handle
}

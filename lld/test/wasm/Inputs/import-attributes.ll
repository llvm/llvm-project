target triple = "wasm32-unknown-unknown"

define void @call_foo() {
  call void @foo();
  ret void
}

declare void @foo() #0

attributes #0 = { "wasm-import-module"="baz" }

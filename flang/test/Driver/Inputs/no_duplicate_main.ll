; Create the symbol 'main'; does not have to be the correct
; signature for 'main', we just need the symbol for the linker
; to fail during the test.

define i32 @main() {
  ret i32 0
}

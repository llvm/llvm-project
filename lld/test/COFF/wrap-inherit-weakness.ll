; REQUIRES: x86

; Check that 'weak' attribute will be inherited to the wrapped symbols.

; RUN: llc %s -mtriple x86_64-mingw -o %t.o --filetype=obj
; RUN: ld.lld -m i386pep -shared -o %t.dll %t.o --entry= --wrap fn

declare extern_weak dso_local void @__real_fn() nounwind
declare dso_local void @fn() nounwind
declare dso_local void @__wrap_fn() nounwind

define dllexport void @caller() nounwind {
  call void @__real_fn()
  call void @fn()
  ret void
}

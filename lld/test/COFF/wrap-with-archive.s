// REQUIRES: x86
// RUN: split-file %s %t.dir
// RUN: llvm-mc -filetype=obj -triple=x86_64-win32-gnu %t.dir/main.s -o %t.main.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-win32-gnu %t.dir/wrap.s -o %t.wrap.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-win32-gnu %t.dir/other.s -o %t.other.obj
// RUN: rm -f %t.lib
// RUN: llvm-ar rcs %t.lib %t.wrap.obj %t.other.obj

// RUN: lld-link -out:%t.exe %t.main.obj -libpath:%T %t.lib -entry:entry -subsystem:console -wrap:foo

// Note: No real definition of foo exists here, but that works fine as long
// as there's no actual references to __real_foo.

#--- main.s
.global entry
entry:
  call foo
  ret

#--- wrap.s
.global __wrap_foo
__wrap_foo:
  call other_func
  ret

#--- other.s
.global other_func
other_func:
  ret

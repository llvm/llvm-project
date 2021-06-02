# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc %t/bar.s -triple=x86_64-apple-macos -filetype=obj -o %t/bar.o
# RUN: %lld -dylib %t/bar.o -o %t/bar.dylib
# RUN: %lld -dylib %t/bar.o -o %t/libbar.dylib
# RUN: %lld -dylib -mark_dead_strippable_dylib %t/bar.o -o %t/bar-strip.dylib

# RUN: llvm-mc %t/foo.s -triple=x86_64-apple-macos -filetype=obj -o %t/foo.o
# RUN: %lld -dylib %t/foo.o -o %t/foo_with_bar.dylib %t/bar.dylib -sub_library bar
# RUN: %lld -dylib %t/foo.o -o %t/foo.dylib

# RUN: llvm-mc %t/weak-foo.s -triple=x86_64-apple-macos -filetype=obj -o %t/weak-foo.o
# RUN: %lld -dylib %t/weak-foo.o -o %t/weak-foo.dylib

# RUN: llvm-mc %t/main.s -triple=x86_64-apple-macos -filetype=obj -o %t/main.o

## foo_with_bar.dylib's reexport should be dropped since it's linked implicitly.
# RUN: %lld -lSystem %t/main.o -o %t/main %t/foo_with_bar.dylib
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOBAR %s
# NOBAR-NOT: bar.dylib
# NOBAR: /usr/lib/libSystem.dylib
# NOBAR-NOT: bar.dylib
# NOBAR: foo_with_bar.dylib
# NOBAR-NOT: bar.dylib

## If bar.dylib is linked explicitly, it should not be dropped.
# RUN: %lld -lSystem %t/main.o -o %t/main %t/foo_with_bar.dylib %t/bar.dylib
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=BAR %s
# BAR: /usr/lib/libSystem.dylib
# BAR: foo_with_bar.dylib
# BAR: bar.dylib

## ...except if -dead-strip_dylibs is passed...
# RUN: %lld -lSystem %t/main.o -o %t/main %t/foo_with_bar.dylib %t/bar.dylib \
# RUN:      -dead_strip_dylibs
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOBAR %s

## ...or bar is explicitly marked dead-strippable.
# RUN: %lld -lSystem %t/main.o -o %t/main %t/foo.dylib %t/bar-strip.dylib
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOBARSTRIP %s
# NOBARSTRIP-NOT: bar-strip.dylib
# NOBARSTRIP: /usr/lib/libSystem.dylib
# NOBARSTRIP-NOT: bar-strip.dylib
# NOBARSTRIP: foo.dylib
# NOBARSTRIP-NOT: bar-strip.dylib

## But -needed_library and -needed-l win over -dead_strip_dylibs again.
# RUN: %lld -lSystem %t/main.o -o %t/main %t/foo_with_bar.dylib \
# RUN:     -needed_library %t/bar.dylib -dead_strip_dylibs
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=BAR %s
# RUN: %lld -lSystem %t/main.o -o %t/main %t/foo_with_bar.dylib \
# RUN:     -L%t -needed-lbar -dead_strip_dylibs
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=BAR %s

## LC_LINKER_OPTION does not count as an explicit reference.
# RUN: llvm-mc %t/linkopt_bar.s -triple=x86_64-apple-macos -filetype=obj -o %t/linkopt_bar.o
# RUN: %lld -lSystem %t/main.o %t/linkopt_bar.o -o %t/main -L %t %t/foo.dylib
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOLIBBAR %s
# NOLIBBAR-NOT: libbar.dylib
# NOLIBBAR: /usr/lib/libSystem.dylib
# NOLIBBAR-NOT: libbar.dylib
# NOLIBBAR: foo.dylib
# NOLIBBAR-NOT: libbar.dylib

## ...but with an additional explicit reference it's not stripped again.
# RUN: %lld -lSystem %t/main.o %t/linkopt_bar.o -o %t/main -L %t %t/foo.dylib -lbar
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=LIBBAR %s
# RUN: %lld -lSystem %t/main.o -o %t/main -L %t %t/foo.dylib -lbar %t/linkopt_bar.o
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=LIBBAR %s
# LIBBAR-DAG: /usr/lib/libSystem.dylib
# LIBBAR-DAG: libbar.dylib
# LIBBAR-DAG: foo.dylib

## Test that a DylibSymbol being replaced by a DefinedSymbol marks the
## dylib as unreferenced.
## (Note: Since there's no dynamic linking in this example, libSystem is
## stripped too. Since libSystem.dylib is needed to run an executable for
## LC_MAIN, the output would crash when run. This matches ld64's behavior (!)
## In practice, every executable uses dynamic linking, which uses
## dyld_stub_binder, which keeps libSystem alive.)
## Test all permutations of (Undefined, Defined, DylibSymbol).
# RUN: %lld -lSystem -dead_strip_dylibs %t/main.o %t/foo.o %t/foo.dylib -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOFOO %s
# RUN: %lld -lSystem -dead_strip_dylibs %t/main.o %t/foo.dylib %t/foo.dylib %t/foo.o -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOFOO %s

# RUN: %lld -lSystem -dead_strip_dylibs %t/foo.o %t/main.o %t/foo.dylib -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOFOO %s
# RUN: %lld -lSystem -dead_strip_dylibs %t/foo.dylib %t/foo.dylib %t/main.o %t/foo.o -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOFOO %s

# RUN: %lld -lSystem -dead_strip_dylibs %t/foo.o %t/foo.dylib %t/main.o -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOFOO %s
# RUN: %lld -lSystem -dead_strip_dylibs %t/foo.dylib %t/foo.dylib %t/foo.o %t/main.o -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOFOO %s
# NOFOO-NOT: foo.dylib

## When linking a weak and a strong symbol from two dylibs, we should keep the
## strong one.
# RUN: %lld -lSystem -dead_strip_dylibs %t/main.o %t/foo.dylib %t/weak-foo.dylib -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOWEAK %s
# RUN: %lld -lSystem -dead_strip_dylibs %t/main.o %t/weak-foo.dylib %t/foo.dylib -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOWEAK %s
# RUN: %lld -lSystem -dead_strip_dylibs %t/foo.dylib %t/weak-foo.dylib %t/main.o -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOWEAK %s
# RUN: %lld -lSystem -dead_strip_dylibs %t/weak-foo.dylib %t/foo.dylib %t/main.o -o %t/main
# RUN: llvm-otool -L %t/main | FileCheck --check-prefix=NOWEAK %s
# NOWEAK-NOT: weak-foo.dylib
# NOWEAK: /foo.dylib
# NOWEAK-NOT: weak-foo.dylib

#--- main.s
.globl _foo, _main
_main:
  callq _foo
  retq

#--- foo.s
.globl _foo
_foo:
  retq

#--- bar.s
.globl _bar
_bar:
  retq

#--- linkopt_bar.s
.linker_option "-lbar"

#--- weak-foo.s
.globl _foo
.weak_definition _foo
_foo:
  retq

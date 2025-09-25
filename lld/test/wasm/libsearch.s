// Based on lld/test/ELF/libsearch.s

// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown \
// RUN:   %p/Inputs/libsearch-dyn.s -o %tdyn.o
// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown \
// RUN:   %p/Inputs/libsearch-st.s -o %tst.o
// RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown \
// RUN:   %p/Inputs/use-bar.s -o %tbar.o
// RUN: mkdir -p %t.dir
// RUN: wasm-ld -shared --experimental-pic %tdyn.o -o %t.dir/libls.so
// RUN: cp -f %t.dir/libls.so %t.dir/libls2.so
// RUN: rm -f %t.dir/libls.a
// RUN: llvm-ar rcs %t.dir/libls.a %tst.o

// Should fail if no library specified
// RUN: not wasm-ld -l 2>&1 \
// RUN:   | FileCheck --check-prefix=NOLIBRARY %s
// NOLIBRARY: -l: missing argument

// Should link normally, because _bar is not used
// RUN: wasm-ld -o %t3 %t.o
// Should not link because of undefined symbol _bar
// RUN: not wasm-ld --no-gc-sections -o /dev/null %t.o %tbar.o 2>&1 \
// RUN:   | FileCheck --check-prefix=UNDEFINED %s
// UNDEFINED: wasm-ld: error: {{.*}}: undefined symbol: _bar

// Should fail if cannot find specified library (without -L switch)
// RUN: not wasm-ld -o /dev/null %t.o -lls 2>&1 \
// RUN:   | FileCheck --check-prefix=NOLIB %s
// NOLIB: unable to find library -lls

// Should use explicitly specified static library
// Also ensure that we accept -L <arg>
// RUN: wasm-ld --emit-relocs --no-gc-sections -o %t3 %t.o -L %t.dir -l:libls.a
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// STATIC: Symbols [
// STATIC: Name: _static

// Should use explicitly specified dynamic library
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -l:libls.so
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s
// DYNAMIC: Symbols [
// DYNAMIC-NOT: Name: _static

// Should prefer static to dynamic when linking regular executable.
// RUN: wasm-ld --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s

// Should prefer dynamic when linking PIE.
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s

// Check for library search order
// RUN: mkdir -p %t.dir2
// RUN: cp %t.dir/libls.a %t.dir2
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir2 -L%t.dir -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s

// -L can be placed after -l
// RUN: wasm-ld -o %t3 %t.o -lls -L%t.dir

// Check long forms as well
// RUN: wasm-ld --emit-relocs --no-gc-sections -o %t3 %t.o --library-path=%t.dir --library=ls
// RUN: wasm-ld --emit-relocs --no-gc-sections -o %t3 %t.o --library-path %t.dir --library ls

// Should not search for dynamic libraries if -Bstatic is specified
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -Bstatic -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// RUN: not wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o /dev/null %t.o -L%t.dir -Bstatic -lls2 2>&1 \
// RUN:   | FileCheck --check-prefix=NOLIB2 %s
// NOLIB2: unable to find library -lls2

// -Bdynamic should restore default behaviour
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -Bstatic -Bdynamic -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s

// -Bstatic and -Bdynamic should affect only libraries which follow them
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -lls -Bstatic -Bdynamic
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -Bstatic -lls -Bdynamic
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s

// Check aliases as well
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -dn -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -non_shared -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -static -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=STATIC %s
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -Bstatic -dy -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s
// RUN: wasm-ld -pie --experimental-pic --emit-relocs --no-gc-sections -o %t3 %t.o -L%t.dir -Bstatic -call_shared -lls
// RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=DYNAMIC %s

/// -r implies -Bstatic and has precedence over -Bdynamic.
// RUN: wasm-ld -r -Bdynamic %t.o -L%t.dir -lls -o %t3.ro
// RUN: llvm-readobj -s -h %t3.ro | FileCheck --check-prefix=RELOCATABLE %s
// RELOCATABLE: Name: _static

.globl _start, _bar
_start:
  .functype _start () -> ()
  end_function

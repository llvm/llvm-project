# REQUIRES: x86

## Regression test for export names containing a 0xff byte, which as a signed
## char equals -1, the value the trie builder uses for end-of-string. _a sits
## right before the 0xff in _a\xffb, and _c\xff ends with it. 0x80 is the most
## negative signed char and shares its _d prefix with an ASCII name.

## llvm-mc does not interpret escapes in quoted symbol names, so printf
## produces the raw bytes.
# RUN: printf '.globl "_a", "_a\377b", "_c\377", "_d\200x", "_dy"\n_a:\n"_a\377b":\n"_c\377":\n"_d\200x":\n_dy:\nret\n' | \
# RUN:   llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t.o
# RUN: %lld -dylib %t.o -o %t.dylib
# RUN: llvm-objdump --macho --exports-trie %t.dylib > %t.txt

## cat -v prints 0xff as M-^? and 0x80 as M-^@, so the checks below can name
## the bytes in plain ASCII. (lit's builtin cat only does this for files, not
## stdin, hence %t.txt.)
# RUN: cat -v %t.txt | FileCheck %s

# CHECK-LABEL: Exports trie:
# CHECK-DAG: _a{{$}}
# CHECK-DAG: _aM-^?b{{$}}
# CHECK-DAG: _cM-^?{{$}}
# CHECK-DAG: _dM-^@x{{$}}
# CHECK-DAG: _dy{{$}}

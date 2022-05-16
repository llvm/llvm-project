// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

#ifdef UNDEFINED
#id
#ifd
#ifde
#elf
#elsif
#elseif
#elfidef
#elfindef
#elfinndef
#els
#endi
#endif

#ifdef UNDEFINED
# in in order to perform
#endif

#ifdef UNDEFINED
#i
#endif

#if special_compiler
#special_compiler_directive
#endif

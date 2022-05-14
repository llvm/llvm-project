// RUN: %clang_cc1 -fsyntax-only -verify=pre-c2x-cpp2b %s
// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify=c2x-cpp2b %s
// RUN: %clang_cc1 -x c++ -std=c++2b -fsyntax-only -verify=c2x-cpp2b %s

// id:        pre-c2x-cpp2b-warning@+12 {{invalid preprocessing directive, did you mean '#if'?}}
// ifd:       pre-c2x-cpp2b-warning@+12 {{invalid preprocessing directive, did you mean '#if'?}}
// ifde:      pre-c2x-cpp2b-warning@+12 {{invalid preprocessing directive, did you mean '#ifdef'?}}
// elf:       pre-c2x-cpp2b-warning@+12 {{invalid preprocessing directive, did you mean '#elif'?}}
// elsif:     pre-c2x-cpp2b-warning@+12 {{invalid preprocessing directive, did you mean '#elif'?}}
// elseif:    pre-c2x-cpp2b-warning@+12 {{invalid preprocessing directive, did you mean '#elif'?}}
// elfidef:   not suggested to '#elifdef'
// elfindef:  not suggested to '#elifdef'
// elfinndef: not suggested to '#elifndef'
// els:       pre-c2x-cpp2b-warning@+12 {{invalid preprocessing directive, did you mean '#else'?}}
// endi:      pre-c2x-cpp2b-warning@+12 {{invalid preprocessing directive, did you mean '#endif'?}}
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
// id:        c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#if'?}}
// ifd:       c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#if'?}}
// ifde:      c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#ifdef'?}}
// elf:       c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#elif'?}}
// elsif:     c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#elif'?}}
// elseif:    c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#elif'?}}
// elfidef:   c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#elifdef'?}}
// elfindef:  c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#elifdef'?}}
// elfinndef: c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#elifndef'?}}
// els:       c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#else'?}}
// endi:      c2x-cpp2b-warning@-12 {{invalid preprocessing directive, did you mean '#endif'?}}

#ifdef UNDEFINED
#i // no diagnostic
#endif

#if special_compiler
#special_compiler_directive // no diagnostic
#endif

// RUN: %clang_cc1 -fsyntax-only -verify %s

// Note that this puts the expected lines before the directives to work around
// limitations in the -verify mode.

/* expected-warning {{missing '(' after '#pragma pack'}}*/ #pragma pack 10
#pragma pack()
#pragma pack(8)

/*expected-warning {{unknown action for '#pragma pack'}}*/ #pragma pack(hello) 
#pragma pack(push)
#pragma pack(pop)

/* expected-warning {{expected integer or identifier in '#pragma pack'}}*/ #pragma pack(push,)
/* expected-warning {{expected integer or identifier in '#pragma pack'}}*/ #pragma pack(push,)
/* expected-warning {{expected integer or identifier in '#pragma pack'}}*/  #pragma pack(pop,) 

#pragma pack(push,i)
/* expected-warning {{expected integer or identifier in '#pragma pack'}}*/ #pragma pack(push,i, 
/* expected-warning {{expected integer or identifier in '#pragma pack'}}*/ #pragma pack(push,i,) 
/* expected-warning {{expected integer or identifier in '#pragma pack'}}*/ #pragma pack(push,i,help) 

#pragma pack(push,8)
/* expected-warning {{missing ')' after '#pragma pack'}}*/ #pragma pack(push,8, 
/* expected-warning {{missing ')' after '#pragma pack'}}*/ #pragma pack(push,8,) 
/* expected-warning {{missing ')' after '#pragma pack'}}*/ #pragma pack(push,i,8 
#pragma pack(push,i,8)

/* expected-warning {{missing ')' after '#pragma pack'}}*/ #pragma pack(push 

_Pragma("pack(push)")
/* expected-warning {{expected integer or identifier in '#pragma pack'}}*/ _Pragma("pack(push,)") 

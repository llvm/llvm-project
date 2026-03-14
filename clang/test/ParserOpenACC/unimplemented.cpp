// RUN: %clang_cc1 %s -verify -fopenacc

// Parser::ParseExternalDeclaration
// expected-error@+2{{invalid OpenACC directive 'havent'}}
// expected-error@+1{{invalid OpenACC clause 'implemented'}}
#pragma acc havent implemented
int foo;

struct S {
// Parser::ParseCXXClassMemberDeclarationWithPragmas
// expected-error@+2{{invalid OpenACC directive 'havent'}}
// expected-error@+1{{invalid OpenACC clause 'implemented'}}
#pragma acc havent implemented
  int foo;
};

void func() {
// Parser::ParseStmtOrDeclarationAfterAttributes
// expected-error@+2{{invalid OpenACC directive 'havent'}}
// expected-error@+1{{invalid OpenACC clause 'implemented'}}
#pragma acc havent implemented
  while(false) {}
}

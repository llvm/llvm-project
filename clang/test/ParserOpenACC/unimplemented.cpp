// RUN: %clang_cc1 %s -verify -fopenacc

// Parser::ParseExternalDeclaration
// expected-error@+1{{invalid OpenACC directive 'not'}}
#pragma acc not yet implemented
int foo;

struct S {
// Parser::ParseCXXClassMemberDeclarationWithPragmas
// expected-error@+1{{invalid OpenACC directive 'not'}}
#pragma acc not yet implemented
  int foo;
};

void func() {
// Parser::ParseStmtOrDeclarationAfterAttributes
// expected-error@+1{{invalid OpenACC directive 'not'}}
#pragma acc not yet implemented
  while(false) {}
}

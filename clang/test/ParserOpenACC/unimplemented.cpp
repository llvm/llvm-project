// RUN: %clang_cc1 %s -verify -fopenacc

// Parser::ParseExternalDeclaration
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc not yet implemented
int foo;

struct S {
// Parser::ParseCXXClassMemberDeclarationWithPragmas
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc not yet implemented
  int foo;
};

void func() {
// Parser::ParseStmtOrDeclarationAfterAttributes
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc not yet implemented
  while(false) {}
}

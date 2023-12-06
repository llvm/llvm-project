// RUN: %clang_cc1 %s -verify -fopenacc

// Parser::ParseExternalDeclaration
// expected-error@+3{{invalid OpenACC directive 'havent'}}
// expected-warning@+2{{OpenACC clause parsing not yet implemented}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc havent implemented
int foo;

struct S {
// Parser::ParseCXXClassMemberDeclarationWithPragmas
// expected-error@+3{{invalid OpenACC directive 'havent'}}
// expected-warning@+2{{OpenACC clause parsing not yet implemented}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc havent implemented
  int foo;
};

void func() {
// Parser::ParseStmtOrDeclarationAfterAttributes
// expected-error@+3{{invalid OpenACC directive 'havent'}}
// expected-warning@+2{{OpenACC clause parsing not yet implemented}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc havent implemented
  while(false) {}
}

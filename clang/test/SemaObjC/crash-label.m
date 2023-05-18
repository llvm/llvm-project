// RUN: %clang_cc1  -fsyntax-only -verify %s

  - (NSDictionary*) _executeScript:(NSString *)source {  // expected-error 2 {{expected a type}} \
                                                         // expected-error {{missing context for method declaration}}
Exit:  [nilArgs release];
}
- (NSDictionary *) _setupKernelStandardMode:(NSString *)source { // expected-error 2 {{expected a type}} \
                                                                 // expected-error {{missing context for method declaration}}
  Exit:   if(_ciKernel && !success ) {

// RUN: %clang %s -S -m64 -o -

@class NSString;


NSString *S = @"\xff\xff___WAIT___"; // expected-warning {{input conversion stopped due to an input byte that does not belong to the input codeset UTF-8}}

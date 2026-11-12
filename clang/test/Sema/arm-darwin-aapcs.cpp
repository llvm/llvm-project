// RUN: %clang_cc1 %s -triple thumbv7-apple-ios -target-abi aapcs -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple thumbv7m-apple-macho -target-abi aapcs -verify -fsyntax-only
// expected-no-diagnostics

// ARM's AAPCS normally has size_t defined as unsigned int, but on Darwin
// some embedded targets use AAPCS with the iOS header files, which define
// size_t as unsigned long.  Make sure that works.
typedef unsigned long size_t;
void* malloc(size_t);
void* operator new(size_t size)
{
  return (malloc(size));
}

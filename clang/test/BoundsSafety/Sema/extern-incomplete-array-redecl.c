
// RUN: %clang_cc1 -isystem %S/mock-sdk -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -isystem %S/mock-sdk -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
#include <extern-array-mock.h>

// expected-note@extern-array-mock.h:3 {{'externArray' declared here}}
// expected-error@+1{{conflicting '__counted_by' attribute with the previous variable declaration}}
extern unsigned externArray[__counted_by(10)];

extern const unsigned externCountedArray[__counted_by(2)];
const unsigned externCountedArray[] = {1, 2};

extern const unsigned externCountVar;
extern const unsigned externCountedArrayVar[__counted_by(externCountVar)];
const unsigned externCountVar = 2;
const unsigned externCountedArrayVar[] = {1, 2};

extern const unsigned externCountVar2;
extern const unsigned externCountedArrayVar2[__counted_by(externCountVar2)];
const unsigned externCountedArrayVar2[] = {1, 2};
const unsigned externCountVar2 = 2;

extern const unsigned externCountVar3;
extern const unsigned externCountedArrayVar3[__counted_by(externCountVar3)];
const unsigned externCountedArrayVar3[] = {1, 2};
const unsigned externCountVar3 = 3; // rdar://129246717 this should be an error

extern const unsigned externCountVar4;
extern const unsigned externCountedArrayVar4[__counted_by(externCountVar4)];
const unsigned externCountVar4 = 3; // rdar://129246717 this should be an error
const unsigned externCountedArrayVar4[] = {1, 2};

extern const unsigned externCountVar5;
extern const unsigned externCountedArrayVar5[__counted_by(externCountVar5)];
const unsigned externCountedArrayVar5[] = {1, 2}; // rdar://129246717 this should be an error

extern const unsigned externCountVar6;
extern const unsigned externCountedArrayVar6[__counted_by(externCountVar6)];
const unsigned externCountVar6 = 3; // rdar://129246717 this should be an error

extern const unsigned externCountedArrayUnsafe[__counted_by(10)];
const unsigned externCountedArrayUnsafe[] = {1, 2}; // rdar://129246717 this should be an error

void bar(const unsigned	*pointer);

void foo(void){
	f();
	bar(externArray);
	unsigned *ptr = externArray;
}

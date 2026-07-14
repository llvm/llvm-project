
// RUN: %clang_cc1 -triple arm64-apple-macos -target-feature +sve -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -triple arm64-apple-macos -target-feature +sve -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int len;

//------------------------------------------------------------------------------
// void pointer
//------------------------------------------------------------------------------
// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
void *__counted_by(len) voidPtr;

// expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
void *__counted_by_or_null(len) voidPtrOrNull;

//------------------------------------------------------------------------------
// Function pointer
//------------------------------------------------------------------------------
typedef int (*fptr_t)(int*);

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'int (int *__single)' is a function type}}
fptr_t __counted_by(len) fPtr;

// expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'int (int *__single)' is a function type}}
fptr_t __counted_by_or_null(len) fPtrOrNull;


//------------------------------------------------------------------------------
// variable length array
//------------------------------------------------------------------------------
struct vla {
    int size;
    char data[__counted_by(size)];
};
typedef struct vla vla_t;

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'vla_t' (aka 'struct vla') is a struct type with a flexible array member}}
vla_t* __counted_by(len) vlaPtr;

// expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'struct vla' is a struct type with a flexible array member}}
struct vla* __counted_by(len) vlaPtr2;

//------------------------------------------------------------------------------
// builtins
//------------------------------------------------------------------------------
#ifdef  __aarch64__
    // CHECK: fix-it:"{{.+}}":{[[@LINE+2]]:17-[[@LINE+2]]:29}:"__sized_by"
    // expected-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because '__SVInt8_t' is a sizeless type}}
    __SVInt8_t* __counted_by(len) countSVInt8;
#endif

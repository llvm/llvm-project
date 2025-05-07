
// RUN: %clang_cc1 -triple arm64-apple-macos -target-feature +sve -fsyntax-only -fbounds-safety -verify %s
// RUN: not %clang_cc1 -triple arm64-apple-macos -target-feature +sve -fsyntax-only -fbounds-safety  -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macos -target-feature +sve -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: not %clang_cc1 -triple arm64-apple-macos -target-feature +sve -fsyntax-only -fbounds-safety  -x objective-c -fexperimental-bounds-safety-objc -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

int len;

//------------------------------------------------------------------------------
// void pointer
//------------------------------------------------------------------------------
// CHECK: fix-it:"{{.+}}":{[[@LINE+2]]:7-[[@LINE+2]]:19}:"__sized_by"
// expected-error@+1{{cannot apply '__counted_by' attribute to 'void *' because 'void' has unknown size; did you mean to use '__sized_by' instead?}}
void *__counted_by(len) voidPtr;

// CHECK: fix-it:"{{.+}}":{[[@LINE+2]]:7-[[@LINE+2]]:27}:"__sized_by_or_null"
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to 'void *' because 'void' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
void *__counted_by_or_null(len) voidPtrOrNull;

//------------------------------------------------------------------------------
// Function pointer
//------------------------------------------------------------------------------
typedef int (*fptr_t)(int*);

// CHECK: fix-it:"{{.+}}":{[[@LINE+2]]:8-[[@LINE+2]]:20}:"__sized_by"
// expected-error@+1{{cannot apply '__counted_by' attribute to 'int (*)(int *__single)' because 'int (int *__single)' has unknown size; did you mean to use '__sized_by' instead?}}
fptr_t __counted_by(len) fPtr;

// CHECK: fix-it:"{{.+}}":{[[@LINE+2]]:8-[[@LINE+2]]:28}:"__sized_by_or_null"
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to 'int (*)(int *__single)' because 'int (int *__single)' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
fptr_t __counted_by_or_null(len) fPtrOrNull;


//------------------------------------------------------------------------------
// variable length array
//------------------------------------------------------------------------------
struct vla {
    int size;
    char data[__counted_by(size)];
};
typedef struct vla vla_t;

// CHECK: fix-it:"{{.+}}":{[[@LINE+2]]:8-[[@LINE+2]]:20}:"__sized_by"
// expected-error@+1{{cannot apply '__counted_by' attribute to 'vla_t *' (aka 'struct vla *') because 'vla_t' (aka 'struct vla') has unknown size; did you mean to use '__sized_by' instead?}}
vla_t* __counted_by(len) vlaPtr;

// CHECK: fix-it:"{{.+}}":{[[@LINE+2]]:13-[[@LINE+2]]:25}:"__sized_by"
// expected-error@+1{{cannot apply '__counted_by' attribute to 'struct vla *' because 'struct vla' has unknown size; did you mean to use '__sized_by' instead?}}
struct vla* __counted_by(len) vlaPtr2;

//------------------------------------------------------------------------------
// builtins
//------------------------------------------------------------------------------
#ifdef  __aarch64__
    // CHECK: fix-it:"{{.+}}":{[[@LINE+2]]:17-[[@LINE+2]]:29}:"__sized_by"
    // expected-error@+1{{cannot apply '__counted_by' attribute to '__SVInt8_t *' because '__SVInt8_t' has unknown size; did you mean to use '__sized_by' instead?}}
    __SVInt8_t* __counted_by(len) countSVInt8;
#endif

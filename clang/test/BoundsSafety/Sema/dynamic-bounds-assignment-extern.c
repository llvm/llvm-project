

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

struct s_t;
extern unsigned size;
extern struct s_t * __sized_by(size) buf;

unsigned size;
struct s_t *__sized_by(size) buf;
// expected-error@+1{{assignment to 'struct s_t *__single __sized_by(size)' (aka 'struct s_t *__single') 'buf' requires corresponding assignment to 'size'; add self assignment 'size = size' if the value has not changed}}
void assign_to_sized_by(struct s_t *__indexable p) { buf = p; }

struct s_t;
extern struct s_t *end;
extern struct s_t * __ended_by(end) start;

struct s_t *end;
struct s_t *__ended_by(end) start;

// expected-error@+1{{assignment to 'struct s_t *__single __ended_by(end)' (aka 'struct s_t *__single') 'start' requires corresponding assignment to 'end'; add self assignment 'end = end' if the value has not changed}}
void assign_to_ended_by(struct s_t *__indexable p) { start = p; }
// expected-error@+1{{assignment to 'struct s_t *__single __ended_by(end)' (aka 'struct s_t *__single') 'end' requires corresponding assignment to 'start'; add self assignment 'start = start' if the value has not changed}}
void assign_to_end(struct s_t *__indexable p) { end = p; }

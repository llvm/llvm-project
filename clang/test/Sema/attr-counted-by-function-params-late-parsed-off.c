// RUN: %clang_cc1 -DNEEDS_LATE_PARSING -fno-experimental-late-parse-attributes -fsyntax-only -verify %s
// RUN: %clang_cc1 -DNEEDS_LATE_PARSING -fsyntax-only -verify %s

// RUN: %clang_cc1 -UNEEDS_LATE_PARSING -fno-experimental-late-parse-attributes -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -UNEEDS_LATE_PARSING -fsyntax-only -verify=ok %s

// On a parameter the argument is always parsed at the end of the prototype, so
// the language option decides only whether a count declared *after* the
// annotated parameter may be named.

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

#ifdef NEEDS_LATE_PARSING

// expected-error@+1{{'counted_by' argument 'count' is declared after the annotated pointer; referring to a later parameter requires '-fexperimental-late-parse-attributes'}}
void fwd_ref(int *__counted_by(count) buf, int count);

// expected-error@+1{{'counted_by_or_null' argument 'count' is declared after the annotated pointer; referring to a later parameter requires '-fexperimental-late-parse-attributes'}}
void fwd_ref_or_null(int *__counted_by_or_null(count) buf, int count);

// expected-error@+1{{'sized_by' argument 'count' is declared after the annotated pointer; referring to a later parameter requires '-fexperimental-late-parse-attributes'}}
void fwd_ref_sized(void *__sized_by(count) buf, int count);

// expected-error@+1{{'sized_by_or_null' argument 'count' is declared after the annotated pointer; referring to a later parameter requires '-fexperimental-late-parse-attributes'}}
void fwd_ref_sized_or_null(void *__sized_by_or_null(count) buf, int count);

// A count in a nested prototype is a forward reference just the same.
// expected-error@+1{{'counted_by' argument 'len' is declared after the annotated pointer; referring to a later parameter requires '-fexperimental-late-parse-attributes'}}
void inner_fwd_ref(void (*cb)(int *__counted_by(len) p, int len));

// Decided by parameter order, not source location: one macro expansion gives
// the attribute and the count the same expansion location.
#define PARAMS int *__counted_by(count) buf, int count
// expected-error@+1{{'counted_by' argument 'count' is declared after the annotated pointer; referring to a later parameter requires '-fexperimental-late-parse-attributes'}}
void macro_fwd_ref(PARAMS);

// The same, written after the declarator.
#define PARAMS_TRAILING int *buf __counted_by(count), int count
// expected-error@+1{{'counted_by' argument 'count' is declared after the annotated pointer; referring to a later parameter requires '-fexperimental-late-parse-attributes'}}
void macro_fwd_ref_trailing(PARAMS_TRAILING);

#else

// ok-no-diagnostics

// A count already in scope needs nothing from the language option.
void back_ref(int count, int *__counted_by(count) buf);
void back_ref_or_null(int count, int *__counted_by_or_null(count) buf);
void back_ref_sized(int count, void *__sized_by(count) buf);
void back_ref_sized_or_null(int count, void *__sized_by_or_null(count) buf);

// The count and the annotated pointer may be separated by other parameters.
void back_ref_interleaved(int count, int other, int *__counted_by(count) buf);

// Several parameters may share one count.
void two_buffers(int count, int *__counted_by(count) a,
                 int *__counted_by(count) b);

// A nested prototype's own earlier parameter is in scope for the same reason.
void inner_back_ref(void (*cb)(int len, int *__counted_by(len) p));

// So is an enclosing prototype's earlier parameter: that clause is still being
// parsed, so its scope holds only what precedes the callback.
void outer_back_ref(int n, void (*cb)(int *__counted_by(n) p));

// A macro-built list is accepted for the same reason: the count still precedes.
#define PARAMS_OK int count, int *__counted_by(count) buf
void macro_back_ref(PARAMS_OK);

#define PARAMS_OK_TRAILING int count, int *buf __counted_by(count)
void macro_back_ref_trailing(PARAMS_OK_TRAILING);

// A definition, not just a prototype.
int sum(int count, int *__counted_by(count) buf) {
  int total = 0;
  for (int i = 0; i < count; ++i)
    total += buf[i];
  return total;
}

#endif

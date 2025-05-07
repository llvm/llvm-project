
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

// expected-note@+8{{passing argument to parameter 'out_start' here}}
// expected-note@+7{{passing argument to parameter 'out_start' here}}
// expected-note@+6{{passing argument to parameter 'out_start' here}}
// expected-note@+5{{passing argument to parameter 'out_start' here}}
// expected-note@+4{{passing argument to parameter 'out_end' here}}
// expected-note@+3{{passing argument to parameter 'out_end' here}}
// expected-note@+2{{passing argument to parameter 'out_end' here}}
// expected-note@+1{{passing argument to parameter 'out_end' here}}
void callee_out_ranges(int *__ended_by(*out_end) *out_start, int **out_end);

void caller_out_ranges_wrong_args(int *__ended_by(*out_end) *out_start, int **out_end) {
  callee_out_ranges(out_start, out_end);
  // XXX: rdar://97038292
  // expected-warning@+2{{incompatible pointer types passing 'int *__single /* __started_by(*out_start) */ ' (aka 'int *__single') to parameter of type 'int *__single /* __started_by(*out_start) */ *__single' (aka 'int *__single*__single'); remove *}}
  // expected-error@+1{{type of 'out_end', 'int *__single /* __started_by(*out_start) */ *__single' (aka 'int *__single*__single'), is incompatible with parameter of type 'int *__single /* __started_by(*out_start) */ ' (aka 'int *__single')}}
  callee_out_ranges(out_start, *out_end);
  // expected-warning@+3{{incompatible pointer types passing 'int *__single __ended_by(*out_end)' (aka 'int *__single') to parameter of type 'int *__single __ended_by(*out_end)*__single' (aka 'int *__single*__single'); remove *}}
  // expected-warning@+2{{incompatible pointer types passing 'int *__single /* __started_by(*out_start) */ ' (aka 'int *__single') to parameter of type 'int *__single /* __started_by(*out_start) */ *__single' (aka 'int *__single*__single'); remove *}}
  // expected-error@+1{{type of 'out_start', 'int *__single __ended_by(*out_end)*__single' (aka 'int *__single*__single'), is incompatible with parameter of type 'int *__single __ended_by(*out_end)' (aka 'int *__single')}}
  callee_out_ranges(*out_start, *out_end);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __ended_by(*out_end)' (aka 'int *__single') to parameter of type 'int *__single __ended_by(*out_end)*__single' (aka 'int *__single*__single'); remove *}}
  // expected-error@+1{{type of 'out_start', 'int *__single __ended_by(*out_end)*__single' (aka 'int *__single*__single'), is incompatible with parameter of type 'int *__single __ended_by(*out_end)' (aka 'int *__single')}}
  callee_out_ranges(*out_start, out_end);
  // expected-error@+1{{type of 'out_end', 'int *__single /* __started_by(*out_start) */ ' (aka 'int *__single'), is incompatible with parameter of type 'int *__single __ended_by(*out_end)' (aka 'int *__single')}}
  callee_out_ranges(out_end, out_start);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __ended_by(*out_end)' (aka 'int *__single') to parameter of type 'int *__single /* __started_by(*out_start) */ *__single' (aka 'int *__single*__single'); remove *}}
  // expected-error@+1{{type of 'out_end', 'int *__single /* __started_by(*out_start) */ ' (aka 'int *__single'), is incompatible with parameter of type 'int *__single __ended_by(*out_end)' (aka 'int *__single')}}
  callee_out_ranges(out_end, *out_start);
  // expected-warning@+3{{incompatible pointer types passing 'int *__single /* __started_by(*out_start) */ ' (aka 'int *__single') to parameter of type 'int *__single __ended_by(*out_end)*__single' (aka 'int *__single*__single'); remove *}}
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __ended_by(*out_end)' (aka 'int *__single') to parameter of type 'int *__single /* __started_by(*out_start) */ *__single' (aka 'int *__single*__single'); remove *}}
  // expected-error@+1{{type of 'out_end', 'int *__single /* __started_by(*out_start) */ *__single' (aka 'int *__single*__single'), is incompatible with parameter of type 'int *__single __ended_by(*out_end)' (aka 'int *__single')}}
  callee_out_ranges(*out_end, *out_start);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single /* __started_by(*out_start) */ ' (aka 'int *__single') to parameter of type 'int *__single __ended_by(*out_end)*__single' (aka 'int *__single*__single'); remove *}}
  // expected-error@+1{{type of 'out_end', 'int *__single /* __started_by(*out_start) */ *__single' (aka 'int *__single*__single'), is incompatible with parameter of type 'int *__single __ended_by(*out_end)' (aka 'int *__single')}}
  callee_out_ranges(*out_end, out_start);
}

// expected-note@+8{{passing argument to parameter 'start' here}}
// expected-note@+7{{passing argument to parameter 'start' here}}
// expected-note@+6{{passing argument to parameter 'start' here}}
// expected-note@+5{{passing argument to parameter 'start' here}}
// expected-note@+4{{passing argument to parameter 'end' here}}
// expected-note@+3{{passing argument to parameter 'end' here}}
// expected-note@+2{{passing argument to parameter 'end' here}}
// expected-note@+1{{passing argument to parameter 'end' here}}
void callee_in_ranges(int *__ended_by(end) start, int *end);

void caller_in_ranges_wrong_args(int *__ended_by(end) start, int *end) {
  callee_in_ranges(start, end);
  // expected-warning@+3{{incompatible pointer types passing 'int *__single __ended_by(end)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single __ended_by(end)' (aka 'int *__single'); remove &}}
  // expected-warning@+2{{incompatible pointer types passing 'int *__single /* __started_by(start) */ *__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single /* __started_by(start) */ ' (aka 'int *__single'); remove &}}
  // expected-error@+1{{type of 'start', 'int *__single __ended_by(end)' (aka 'int *__single'), is incompatible with parameter of type 'int *__single __ended_by(end)' (aka 'int *__single')}}
  callee_in_ranges(&start, &end);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __ended_by(end)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single __ended_by(end)' (aka 'int *__single'); remove &}}
  // expected-error@+1{{type of 'start', 'int *__single __ended_by(end)' (aka 'int *__single'), is incompatible with parameter of type 'int *__single __ended_by(end)' (aka 'int *__single')}}
  callee_in_ranges(&start, end);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single /* __started_by(start) */ *__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single /* __started_by(start) */ ' (aka 'int *__single'); remove &}}
  // expected-error@+1{{type of 'end', 'int *__single /* __started_by(start) */ ' (aka 'int *__single'), is incompatible with parameter of type 'int *__single /* __started_by(start) */ ' (aka 'int *__single')}}
  callee_in_ranges(start, &end);
  callee_in_ranges(end, start);
  // expected-warning@+3{{incompatible pointer types passing 'int *__single /* __started_by(start) */ *__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single __ended_by(end)' (aka 'int *__single'); remove &}}
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __ended_by(end)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single /* __started_by(start) */ ' (aka 'int *__single'); remove &}}
  // expected-error@+1{{type of 'end', 'int *__single /* __started_by(start) */ ' (aka 'int *__single'), is incompatible with parameter of type 'int *__single __ended_by(end)' (aka 'int *__single')}}
  callee_in_ranges(&end, &start);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single /* __started_by(start) */ *__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single __ended_by(end)' (aka 'int *__single'); remove &}}
  // expected-error@+1{{type of 'end', 'int *__single /* __started_by(start) */ ' (aka 'int *__single'), is incompatible with parameter of type 'int *__single __ended_by(end)' (aka 'int *__single')}}
  callee_in_ranges(&end, start);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __ended_by(end)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single /* __started_by(start) */ ' (aka 'int *__single'); remove &}}
  // expected-error@+1{{type of 'start', 'int *__single __ended_by(end)' (aka 'int *__single'), is incompatible with parameter of type 'int *__single /* __started_by(start) */ ' (aka 'int *__single')}}
  callee_in_ranges(end, &start);
}
// expected-note@+9{{passing argument to parameter 'out_count' here}}
// expected-note@+8{{passing argument to parameter 'out_count' here}}
// expected-note@+7{{passing argument to parameter 'out_count' here}}
// expected-note@+6{{passing argument to parameter 'out_buf' here}}
// expected-note@+5{{passing argument to parameter 'out_buf' here}}
// expected-note@+4{{passing argument to parameter 'out_buf' here}}
// expected-note@+3{{passing argument to parameter 'out_buf' here}}
// expected-note@+2{{passing argument to parameter 'out_buf' here}}
// expected-note@+1{{passing argument to parameter 'out_buf' here}}
void callee_out_count_buf(int *__counted_by(*out_count) *out_buf, int *out_count);

void caller_out_count_wrong_args(int *__counted_by(*out_count) *out_buf, int *out_count) {
  callee_out_count_buf(out_buf, out_count);
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  callee_out_count_buf(out_buf, *out_count);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __counted_by(*out_count)' (aka 'int *__single') to parameter of type 'int *__single __counted_by(*out_count)*__single' (aka 'int *__single*__single'); remove *}}
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  callee_out_count_buf(*out_buf, *out_count);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __counted_by(*out_count)' (aka 'int *__single') to parameter of type 'int *__single __counted_by(*out_count)*__single' (aka 'int *__single*__single'); remove *}}
  // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single __counted_by(*out_count)*__single' (aka 'int *__single*__single')}}
  callee_out_count_buf(*out_buf, out_count);
  // expected-warning@+3{{incompatible pointer types passing 'int *__single' to parameter of type 'int *__single __counted_by(*out_count)*__single' (aka 'int *__single*__single'); take the address with &}}
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __counted_by(*out_count)*__single' (aka 'int *__single*__single') to parameter of type 'int *__single'; dereference with *}}
  // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single __counted_by(*out_count)*__single' (aka 'int *__single*__single')}}
  callee_out_count_buf(out_count, out_buf);
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  callee_out_count_buf(*out_count, out_buf);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single' to parameter of type 'int *__single __counted_by(*out_count)*__single' (aka 'int *__single*__single'); take the address with &}}
  // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single __counted_by(*out_count)*__single' (aka 'int *__single*__single')}}
  callee_out_count_buf(out_count, *out_buf);
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  callee_out_count_buf(*out_count, *out_buf);
}

// expected-note@+8{{passing argument to parameter 'count' here}}
// expected-note@+7{{passing argument to parameter 'count' here}}
// expected-note@+6{{passing argument to parameter 'count' here}}
// expected-note@+5{{passing argument to parameter 'buf' here}}
// expected-note@+4{{passing argument to parameter 'buf' here}}
// expected-note@+3{{passing argument to parameter 'buf' here}}
// expected-note@+2{{passing argument to parameter 'buf' here}}
// expected-note@+1{{passing argument to parameter 'buf' here}}
void callee_in_count_buf(int *__counted_by(count) buf, int count);

void caller_in_count_wrong_args(int *__counted_by(count) buf, int count) {
  callee_in_count_buf(buf, count);
  // expected-warning@+3{{incompatible pointer types passing 'int *__single __counted_by(count)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single __counted_by(count)' (aka 'int *__single'); remove &}}
  // expected-error@+2{{incompatible pointer to integer conversion passing 'int *__bidi_indexable' to parameter of type 'int'; remove &}}
  // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single __counted_by(count)' (aka 'int *__single')}}
  callee_in_count_buf(&buf, &count);
  // expected-warning@+2{{incompatible pointer types passing 'int *__single __counted_by(count)*__bidi_indexable' (aka 'int *__single*__bidi_indexable') to parameter of type 'int *__single __counted_by(count)' (aka 'int *__single'); remove &}}
  // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int *__single __counted_by(count)' (aka 'int *__single')}}
  callee_in_count_buf(&buf, count);
  // expected-error@+2{{incompatible pointer to integer conversion passing 'int *__bidi_indexable' to parameter of type 'int'; remove &}}
  // expected-error@+1{{passing address of 'count' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  callee_in_count_buf(buf, &count);
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  callee_in_count_buf(count, buf);
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  callee_in_count_buf(count, *buf);
  // expected-error@+2{{incompatible pointer to integer conversion passing 'int *__single __counted_by(count)' (aka 'int *__single') to parameter of type 'int'; dereference with *}}
  // expected-error@+1{{passing address of 'count' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  callee_in_count_buf(&count, buf);
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  callee_in_count_buf(count, &buf);
  // expected-error@+1{{passing address of 'count' referred to by '__counted_by' to a parameter that is not referred to by the same attribute}}
  callee_in_count_buf(&count, *buf);
}

// XXX: rdar://97041755
void callee_vargs(int, ...);

void caller_vargs_with_out_count(int *__counted_by(count) buf, int count) {
  callee_vargs(1, &buf);
  callee_vargs(1, &count);
  callee_vargs(2, &buf, &count);
  callee_vargs(2, buf, &count);
  callee_vargs(2, &buf, count);
  callee_vargs(2, buf, count);
}

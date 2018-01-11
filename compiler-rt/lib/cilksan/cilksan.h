#ifndef __CILKSAN_H__
#define __CILKSAN_H__

#if defined (__cplusplus)
extern "C" {
#endif

int  __cilksan_error_count(void);
void __cilksan_enable_checking(void);
void __cilksan_disable_checking(void);
void __cilksan_begin_reduce_strand(void);
void __cilksan_end_reduce_strand(void);
void __cilksan_begin_update_strand(void);
void __cilksan_end_update_strand(void);

// This funciton parse the input supplied to the user program and get the params
// meant for cilksan (everything after "--").  It return the index in which it
// found "--" so the user program knows when to stop parsing inputs.
int __cilksan_parse_input(int argc, char *argv[]);

#if defined (__cplusplus)
}
#endif

#endif // __CILKSAN_H__

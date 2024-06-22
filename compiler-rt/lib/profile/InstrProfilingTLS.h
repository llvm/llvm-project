#ifndef INSTR_PROFILING_TLS_H
#define INSTR_PROFILING_TLS_H

char *__llvm_profile_begin_tls_counters(void);
char *__llvm_profile_end_tls_counters(void);

/*!
 * \brief Add counter values from TLS to the global counters for the program
 *
 * On thread exit, atomically add the values in TLS counters to the static
 * counters for the whole process.
 */
void __llvm_profile_tls_counters_finalize(void);

/*
 * Dylib stuff
 */
typedef void (*texit_fnc)(void);

typedef struct texit_fn_node {
  struct texit_fn_node *prev;
  texit_fnc fn;
  struct texit_fn_node *next;
} texit_fn_node;

// TODO: really this should be write-preferring rwlocked
struct texit_fn_registry {
  int texit_mtx;
  texit_fn_node head;
  texit_fn_node tail;
};

void register_tls_prfcnts_module_thread_exit_handler(texit_fn_node *new_node);
void unregister_tls_prfcnts_module_thread_exit_handler(texit_fn_node *new_node);
void run_thread_exit_handlers(void);

void register_profile_intercepts();

#endif

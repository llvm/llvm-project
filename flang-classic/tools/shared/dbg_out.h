/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Defines macros and interfaces to the debug information routines
 *
 * Defines the dwarf debug information passed from the Fortran front-end to the
 * Fortran back-end.
 */

extern FILE *dbg_file;
extern char *dbg_file_name;

/* renewed on a function by function basis */
#define DEBUG_FUNC_AREA 17

/* holds the containing subroutine definitions while processing contain'd
 * subroutine.  Renewed when a new top level subroutine is  processed.
 */
#define DEBUG_ENCLOSING_FUNC_AREA 19

/* renewed on a module by module  basis */
#define DEBUG_MODULE_AREA 16

/* kept for the length of the program */
#define DEBUG_GBL_AREA 18

void emit_elf_namelist_member(int);
void create_debug_table(void);
void create_debug_func_table(int debug_area);
void grow_debug_func_table(int newsize, int debug_area);
void create_debug_module_table(void);
void null_module_name(void);

#define OUTPUT_DWARF 0

typedef unsigned short kind_t;

#define TAG_default 0x8fff

/* names of module data/subroutine when USEd */
typedef struct module_altname *mod_altptr;

typedef struct module_altname {
  int sptr;   /* sptr of name , used in front end */
  char *name; /* alternate name, not used in front end */
  mod_altptr next;
} module_altname;

typedef struct ref_symbol {
  int *symnum;
  int size;
  mod_altptr *altname;
} ref_symbol;

/* table of modules and their names

   module information is not saved between modules.  This is what
   we need to keep around: the name, and the dwarf label entry,
   so that subroutines that USE modules can refernce them in
   AT_use
*/
typedef struct mod_label_list_tag *mod_label_list_ptr;

typedef struct mod_label_list_tag {
  char *name;
  int label;
  mod_label_list_ptr next;
  int defined; /* true if defined, false if it is a
                  forward reference (when label > 0)
                */
} mod_label_list;

union a_debug_entry_tag;
typedef union a_debug_entry_tag *a_debug_entry_ptr;

/* entries common to all TAG_variable, TAG_subroutine , etc */
typedef struct common_tag {

  kind_t kind;   /* TAG_ type, if different from
                    the symbol table entry.
                  */
  char *name;    /* demangled name */
  int dwarf_lab; /* When the dwarf gets created,
                    the dwarf lab is stored here.
                    Needed for Namelists, forward
                    references
                  */
  a_debug_entry_ptr next;
  char *mangled_name; /* as seen in the symbol table */
  char *stag_name;    /* scope */
  char *scope_name;   /* scope */
  mod_altptr altname; /* altname */
} a_common_struct;

#define NAMELIST_SIZE 10000
typedef struct a_debug_module_entry_tag {
  a_common_struct common;
#define MOD_LIMIT 200
  int num_used_modules;
  mod_label_list_ptr mod_list_ptr;
  int funcline;

  char *filename;

} a_module_struct;
typedef struct a_debug_subroutine_entry_tag {
  a_common_struct common;
  int num_used_modules;
  mod_label_list_ptr mod_list_ptr;

} a_subroutine_struct;
typedef struct a_debug_variable_entry_tag {
  a_common_struct common;
  int allocated;
  int associated;

} a_variable_struct;

union a_debug_entry_tag {
  a_common_struct common;
  a_module_struct module;
  a_subroutine_struct subroutine;
  a_variable_struct variable;
};

typedef struct a_module_class_entry_tag *a_debug_module_entry_ptr;
typedef a_debug_entry_ptr *debug_info_type;

extern debug_info_type debug_info;
extern debug_info_type enclosing_func_debug_info;
extern debug_info_type debug_module_info;
extern int debug_module_table_size;
extern int enclosing_func_table_size;
extern int debug_func_table_size;

#define f90_dwarf_exists(sptr)                                     \
  ((debug_info && (sptr > -1) && (sptr < debug_func_table_size) && \
    (debug_info[sptr] != 0)))
#define f90_mod_dwarf_exists(sptr)                                         \
  (debug_module_info && (sptr > -1) && (sptr < debug_module_table_size) && \
   debug_module_info[sptr])
extern a_debug_entry_ptr alloc_debug_entry(int, int);

extern a_debug_entry_ptr dwarf_debug_root;
extern void gen_module(char *, a_debug_entry_ptr);
extern void dwarf_set_fn(void); /* dwarf_o.c */

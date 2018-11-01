#include <cstdlib>
#include <assert.h>
#include "csan.h"

#define CSIRT_API __attribute__((visibility("default")))

// ------------------------------------------------------------------------
// Front end data (FED) table structures.
// ------------------------------------------------------------------------

// A FED table is a flat list of FED entries, indexed by a CSI
// ID. Each FED table has its own private ID space.
typedef struct {
  int64_t num_entries;
  csan_source_loc_t *entries;
} fed_table_t;

// Types of FED tables that we maintain across all units.
typedef enum {
  FED_TYPE_FUNCTIONS,
  FED_TYPE_FUNCTION_EXIT,
  FED_TYPE_BASICBLOCK,
  FED_TYPE_CALLSITE,
  FED_TYPE_LOAD,
  FED_TYPE_STORE,
  FED_TYPE_DETACH,
  FED_TYPE_TASK,
  FED_TYPE_TASK_EXIT,
  FED_TYPE_DETACH_CONTINUE,
  FED_TYPE_SYNC,
  FED_TYPE_ALLOCA,
  FED_TYPE_ALLOCFN,
  FED_TYPE_FREE,
  NUM_FED_TYPES // Must be last
} fed_type_t;

// An object table is a flat list of object source-location entries indexed by
// CSI ID.
typedef struct {
  int64_t num_entries;
  obj_source_loc_t *entries;
} obj_table_t;

// Types of object tables that we maintain across all units.
typedef enum {
  OBJ_TYPE_LOAD,
  OBJ_TYPE_STORE,
  OBJ_TYPE_ALLOCA,
  OBJ_TYPE_ALLOCFN,
  NUM_OBJ_TYPES // Must be last
} obj_type_t;

const char *allocfn_str[] =
  {
   "void *malloc(size_t size)",
   "void *valloc(size_t size)",
   "void *calloc(size_t count, size_t size)",
   "void *realloc(void *ptr, size_t size)",
   "void *reallocf(void *ptr, size_t size)",
   "void *operator new(unsigned int)",
   "void *operator new(unsigned int, nothrow)",
   "void *operator new(unsigned long)",
   "void *operator new(unsigned long, nothrow)",
   "void *operator new[](unsigned int)",
   "void *operator new[](unsigned int, nothrow)",
   "void *operator new[](unsigned long)",
   "void *operator new[](unsigned long, nothrow)",
   "void *operator new(unsigned int)",
   "void *operator new(unsigned int, nothrow)",
   "void *operator new(unsigned long long)",
   "void *operator new(unsigned long long, nothrow)",
   "void *operator new[](unsigned int)",
   "void *operator new[](unsigned int, nothrow)",
   "void *operator new[](unsigned long long)",
   "void *operator new[](unsigned long long, nothrow)",
  };

const char *free_str[] =
  {
   "void free(void *ptr)",
   "void operator delete(void*)",
   "void operator delete(void*, nothrow)",
   "void operator delete(void*, unsigned int)",
   "void operator delete(void*, unsigned long)",
   "void operator delete[](void*)",
   "void operator delete[](void*, nothrow)",
   "void operator delete[](void*, unsigned int)",
   "void operator delete[](void*, unsigned long)",
   "void operator delete(void*)",
   "void operator delete(void*, nothrow)",
   "void operator delete(void*, unsigned int)",
   "void operator delete(void*)",
   "void operator delete(void*, nothrow)",
   "void operator delete(void*, unsigned long long)",
   "void operator delete[](void*)",
   "void operator delete[](void*, nothrow)",
   "void operator delete[](void*, unsigned int)",
   "void operator delete[](void*)",
   "void operator delete[](void*, nothrow)",
   "void operator delete[](void*, unsigned long long)"
  };

// ------------------------------------------------------------------------
// Globals
// ------------------------------------------------------------------------

// The list of FED tables. This is indexed by a value of
// 'fed_type_t'.
static fed_table_t *fed_tables = NULL;

// Initially false, set to true once the first unit is initialized,
// which results in the FED list being initialized.
static bool fed_tables_initialized = false;

// The list of object tables. This is indexed by a value of 'obj_type_t'.
static obj_table_t *obj_tables = NULL;

// Initially false, set to true once the first unit is initialized,
// which results in the FED list being initialized.
static bool obj_tables_initialized = false;

// Initially false, set to true once the first unit is initialized,
// which results in the __csi_init() function being called.
static bool csi_init_called = false;

// ------------------------------------------------------------------------
// Private function definitions
// ------------------------------------------------------------------------

// Initialize the FED tables list, indexed by a value of type
// fed_type_t. This is called once, by the first unit to load.
static void initialize_fed_tables() {
  fed_tables = new fed_table_t[NUM_FED_TYPES];
  assert(fed_tables != NULL);
  for (int i = 0; i < (int)NUM_FED_TYPES; ++i) {
    fed_table_t table;
    table.num_entries = 0;
    table.entries = NULL;
    fed_tables[i] = table;
  }
  fed_tables_initialized = true;
}

// Ensure that the FED table of the given type has enough memory
// allocated to add a new unit's entries.
static void ensure_fed_table_capacity(fed_type_t fed_type,
                                      int64_t num_new_entries) {
  if (!fed_tables_initialized)
    initialize_fed_tables();

  assert(num_new_entries >= 0);

  fed_table_t *table = &fed_tables[fed_type];
  int64_t total_num_entries = table->num_entries + num_new_entries;
  if (num_new_entries > 0) {
    if (!table->entries)
      table->entries = new csan_source_loc_t[total_num_entries];
    else {
      csan_source_loc_t *old_entries = table->entries;
      table->entries = new csan_source_loc_t[total_num_entries];
      for (int i = 0; i < table->num_entries; ++i)
        table->entries[i] = old_entries[i];
      delete[] old_entries;
    }
    table->num_entries = total_num_entries;
    assert(table->entries != NULL);
  }
}

// Add a new FED table of the given type.
static inline void add_fed_table(fed_type_t fed_type, int64_t num_entries,
                                 const csan_source_loc_t *fed_entries) {
  ensure_fed_table_capacity(fed_type, num_entries);
  fed_table_t *table = &fed_tables[fed_type];
  csi_id_t base = table->num_entries - num_entries;
  for (csi_id_t i = 0; i < num_entries; ++i)
    table->entries[base + i] = fed_entries[i];
}

// The unit-local counter pointed to by 'fed_id_base' keeps track of
// that unit's "base" ID value of the given type (recall that there is
// a private ID space per FED type). The "base" ID value is the global
// ID that corresponds to the unit's local ID 0. This function stores
// the correct value into a unit's base ID.
static inline void update_ids(fed_type_t fed_type, int64_t num_entries,
                              csi_id_t *fed_id_base) {
  fed_table_t *table = &fed_tables[fed_type];
  // The base ID is the current number of FED entries before adding
  // the new FED table.
  *fed_id_base = table->num_entries - num_entries;
}

// Return the FED entry of the given type, corresponding to the given
// CSI ID.
static inline const csan_source_loc_t *get_fed_entry(fed_type_t fed_type,
                                                     const csi_id_t csi_id) {
  // TODO(ddoucet): threadsafety
  fed_table_t *table = &fed_tables[fed_type];
  if (csi_id < table->num_entries) {
    assert(table->entries != NULL);
    return &table->entries[csi_id];
  }
  return NULL;
}

// Initialize the object tables list, indexed by a value of type
// obj_type_t. This is called once, by the first unit to load.
static void initialize_obj_tables() {
  obj_tables = new obj_table_t[NUM_OBJ_TYPES];
  assert(obj_tables != NULL);
  for (int i = 0; i < (int)NUM_OBJ_TYPES; ++i) {
    obj_table_t table;
    table.num_entries = 0;
    table.entries = NULL;
    obj_tables[i] = table;
  }
  obj_tables_initialized = true;
}

// Ensure that the object table of the given type has enough memory allocated to
// add a new unit's entries.
static void ensure_obj_table_capacity(obj_type_t obj_type,
                                      int64_t num_new_entries) {
  if (!obj_tables_initialized)
    initialize_obj_tables();

  assert(num_new_entries >= 0);

  obj_table_t *table = &obj_tables[obj_type];
  int64_t total_num_entries = table->num_entries + num_new_entries;
  if (num_new_entries > 0) {
    if (!table->entries)
      table->entries = new obj_source_loc_t[total_num_entries];
    else {
      obj_source_loc_t *old_entries = table->entries;
      table->entries = new obj_source_loc_t[total_num_entries];
      for (int i = 0; i < table->num_entries; ++i)
        table->entries[i] = old_entries[i];
      delete[] old_entries;
    }
    table->num_entries = total_num_entries;
    assert(table->entries != NULL);
  }
}

// Add a new object table of the given type.
static inline void add_obj_table(obj_type_t obj_type, int64_t num_entries,
                                 const obj_source_loc_t *obj_entries) {
  ensure_obj_table_capacity(obj_type, num_entries);
  obj_table_t *table = &obj_tables[obj_type];
  csi_id_t base = table->num_entries - num_entries;
  for (csi_id_t i = 0; i < num_entries; ++i)
    table->entries[base + i] = obj_entries[i];
}

// Return the object entry of the given type, corresponding to the given
// CSI ID.
static inline const obj_source_loc_t *get_obj_entry(obj_type_t obj_type,
                                                    const csi_id_t csi_id) {
  // TODO(ddoucet): threadsafety
  obj_table_t *table = &obj_tables[obj_type];
  if (csi_id < table->num_entries) {
    assert(table->entries != NULL);
    return &table->entries[csi_id];
  }
  return NULL;
}

// ------------------------------------------------------------------------
// External function definitions, including CSIRT API functions.
// ------------------------------------------------------------------------

EXTERN_C

void __csan_unit_init(const char * const file_name,
                      const csan_instrumentation_counts_t counts);

// Not used at the moment
// __thread bool __csi_disable_instrumentation;

typedef struct {
  int64_t num_entries;
  csi_id_t *id_base;
  const csan_source_loc_t *entries;
} unit_fed_table_t;

typedef struct {
  int64_t num_entries;
  const obj_source_loc_t *entries;
} unit_obj_table_t;

// Function signature for the function (generated by the CSI compiler
// pass) that updates the callsite to function ID mappings.
typedef void (*__csi_init_callsite_to_functions)();

static inline void compute_inst_counts(csan_instrumentation_counts_t *counts,
                                       unit_fed_table_t *unit_fed_tables) {
  int64_t *base = (int64_t *)counts;
  for (int i = 0; i < NUM_FED_TYPES; i++)
    *(base + i) = unit_fed_tables[i].num_entries;
}

// A call to this is inserted by the CSI compiler pass, and occurs
// before main().
CSIRT_API
void __csirt_unit_init(const char * const name,
                       unit_fed_table_t *unit_fed_tables,
                       unit_obj_table_t *unit_obj_tables,
                       __csi_init_callsite_to_functions callsite_to_func_init) {
  // Make sure we don't instrument things in __csi_init or __csi_unit init.
  // __csi_disable_instrumentation = true;

  // TODO(ddoucet): threadsafety
  if (!csi_init_called) {
    __csi_init();
    csi_init_called = true;
  }

  // Add all FED tables from the new unit
  for (int i = 0; i < NUM_FED_TYPES; ++i) {
    add_fed_table((fed_type_t)i, unit_fed_tables[i].num_entries,
                  unit_fed_tables[i].entries);
    update_ids((fed_type_t)i, unit_fed_tables[i].num_entries,
               unit_fed_tables[i].id_base);
  }

  // Add all object tables from the new unit
  for (int i = 0; i < NUM_OBJ_TYPES; ++i) {
    add_obj_table((obj_type_t)i, unit_obj_tables[i].num_entries,
                  unit_obj_tables[i].entries);
  }

  // Initialize the callsite -> function mappings. This must happen
  // after the base IDs have been updated.
  callsite_to_func_init();

  // Call into the tool implementation.
  csan_instrumentation_counts_t counts;
  compute_inst_counts(&counts, unit_fed_tables);
  __csan_unit_init(name, counts);

  // Reset disable flag.
  // __csi_disable_instrumentation = false;
}

CSIRT_API
const csan_source_loc_t *__csan_get_func_source_loc(const csi_id_t func_id) {
  return get_fed_entry(FED_TYPE_FUNCTIONS, func_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_func_exit_source_loc(
    const csi_id_t func_exit_id) {
  return get_fed_entry(FED_TYPE_FUNCTION_EXIT, func_exit_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_bb_source_loc(const csi_id_t bb_id) {
  return get_fed_entry(FED_TYPE_BASICBLOCK, bb_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_call_source_loc(const csi_id_t call_id) {
  return get_fed_entry(FED_TYPE_CALLSITE, call_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_load_source_loc(const csi_id_t load_id) {
  return get_fed_entry(FED_TYPE_LOAD, load_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_store_source_loc(const csi_id_t store_id) {
  return get_fed_entry(FED_TYPE_STORE, store_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_detach_source_loc(const csi_id_t detach_id) {
  return get_fed_entry(FED_TYPE_DETACH, detach_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_task_source_loc(const csi_id_t task_id) {
  return get_fed_entry(FED_TYPE_TASK, task_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_task_exit_source_loc(
    const csi_id_t task_exit_id) {
  return get_fed_entry(FED_TYPE_TASK_EXIT, task_exit_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_detach_continue_source_loc(
    const csi_id_t detach_continue_id) {
  return get_fed_entry(FED_TYPE_DETACH_CONTINUE, detach_continue_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_sync_source_loc(const csi_id_t sync_id) {
  return get_fed_entry(FED_TYPE_SYNC, sync_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_alloca_source_loc(const csi_id_t alloca_id) {
  return get_fed_entry(FED_TYPE_ALLOCA, alloca_id);
}

CSIRT_API
const csan_source_loc_t *__csan_get_allocfn_source_loc(const csi_id_t allocfn_id) {
  return get_fed_entry(FED_TYPE_ALLOCFN, allocfn_id);
}
CSIRT_API
const csan_source_loc_t *__csan_get_free_source_loc(const csi_id_t free_id) {
  return get_fed_entry(FED_TYPE_FREE, free_id);
}

CSIRT_API
const obj_source_loc_t *__csan_get_load_obj_source_loc(const csi_id_t load_id) {
  return get_obj_entry(OBJ_TYPE_LOAD, load_id);
}

CSIRT_API
const obj_source_loc_t *__csan_get_store_obj_source_loc(const csi_id_t store_id) {
  return get_obj_entry(OBJ_TYPE_STORE, store_id);
}

CSIRT_API
const obj_source_loc_t *__csan_get_alloca_obj_source_loc(const csi_id_t alloca_id) {
  return get_obj_entry(OBJ_TYPE_ALLOCA, alloca_id);
}

CSIRT_API
const obj_source_loc_t *__csan_get_allocfn_obj_source_loc(const csi_id_t allocfn_id) {
  return get_obj_entry(OBJ_TYPE_ALLOCFN, allocfn_id);
}

CSIRT_API
const char *__csan_get_allocfn_str(const allocfn_prop_t prop) {
  return allocfn_str[prop.allocfn_ty];
}

CSIRT_API
const char *__csan_get_free_str(const free_prop_t prop) {
  return free_str[prop.free_ty];
}

EXTERN_C_END

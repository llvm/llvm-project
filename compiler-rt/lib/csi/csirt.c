#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdatomic.h>
#include <csi/csi.h>

// Compile-time assert the property structs are 64 bits.
static_assert(sizeof(func_prop_t) == 8, "Size of func_prop_t is not 64 bits.");
static_assert(sizeof(func_exit_prop_t) == 8,
              "Size of func_exit_prop_t is not 64 bits.");
static_assert(sizeof(bb_prop_t) == 8, "Size of bb_prop_t is not 64 bits.");
static_assert(sizeof(call_prop_t) == 8, "Size of call_prop_t is not 64 bits.");
static_assert(sizeof(load_prop_t) == 8, "Size of load_prop_t is not 64 bits.");
static_assert(sizeof(store_prop_t) == 8,
              "Size of store_prop_t is not 64 bits.");
static_assert(sizeof(alloca_prop_t) == 8,
              "Size of alloca_prop_t is not 64 bits.");
static_assert(sizeof(allocfn_prop_t) == 8,
              "Size of allocfn_prop_t is not 64 bits.");
static_assert(sizeof(free_prop_t) == 8, "Size of free_prop_t is not 64 bits.");

#define CSIRT_API __attribute__((visibility("default")))

// ------------------------------------------------------------------------
// Front end data (FED) table structures.
// ------------------------------------------------------------------------

// A FED table is a flat list of FED entries, indexed by a CSI
// ID. Each FED table has its own private ID space.
typedef struct {
    uint64_t num_entries;
    source_loc_t *entries;
} fed_table_t;

// A FED table index is an array of pointers to equally-sized FED
// tables.
typedef struct {
  uint64_t num_tables;
  uint64_t capacity;
  uint64_t num_total_entries;
  fed_table_t *tables;
} fed_table_index_t;

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

static_assert(sizeof(instrumentation_counts_t) ==
              sizeof(csi_id_t) * NUM_FED_TYPES,
              "Mismatch between NUM_FED_TYPES and size of "
              "instrumentation_counts_t");

// A SizeInfo table is a flat list of SizeInfo entries, indexed by a CSI ID.
typedef struct {
  int64_t num_entries;
  sizeinfo_t *entries;
} sizeinfo_table_t;

// Types of sizeinfo tables that we maintain across all units.
typedef enum {
  SIZEINFO_TYPE_BASICBLOCK,
  NUM_SIZEINFO_TYPES // Must be last
} sizeinfo_type_t;

const char *allocfn_str[] =
  {
   "void *malloc(size_t size)",
   "void *valloc(size_t size)",
   "void *aligned_alloc(align_val_t, size)",
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
   "void *operator new(unsigned int, align_val_t)",
   "void *operator new(unsigned long, align_val_t)",
   "void *operator new[](unsigned int, align_val_t)",
   "void *operator new[](unsigned long, align_val_t)",
   "void *operator new(unsigned int, align_val_t, nothrow)",
   "void *operator new(unsigned long, align_val_t, nothrow)",
   "void *operator new[](unsigned int, align_val_t, nothrow)",
   "void *operator new[](unsigned long, align_val_t, nothrow)",
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
   "void operator delete[](void*, unsigned long long)",
   "void operator delete(void*, align_val_t)",
   "void operator delete(void*, align_val_t, nothrow)",
   "void operator delete[](void*, align_val_t)",
   "void operator delete[](void*, align_val_t, nothrow)",
  };

// ------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------
static const uint64_t NUM_ELEMENTS_PER_TABLE = ((uint64_t)1) << 16;
static const uint64_t DEFAULT_NUM_POINTERS_TO_TABLES = ((uint64_t)1) << 8;

// ------------------------------------------------------------------------
// Globals
// ------------------------------------------------------------------------

// The list of FED table indices. This is indexed by a value of
// 'fed_type_t'.
static fed_table_index_t *fed_tables = NULL;

// Initially false, set to true once the first unit is initialized,
// which results in the FED list being initialized.
static bool fed_tables_initialized = false;

// The list of SizeInfo tables. This is indexed by a value of
// 'sizeinfo_type_t'.
static sizeinfo_table_t *sizeinfo_tables = NULL;

// Initially false, set to true once the first unit is initialized,
// which results in the SizeInfo list being initialized.
static bool sizeinfo_tables_initialized = false;

// Initially false, set to true once the first unit is initialized,
// which results in the __csi_init() function being called.
static bool csi_init_called = false;

// ------------------------------------------------------------------------
// Private function definitions
// ------------------------------------------------------------------------

// NOTE: All functions modifying the FED tables and the index are NOT thread 
// safe and MUST be protected by a mutex.

// Append a table to the index, which is resized if necessary.
static fed_table_t *
allocate_new_table_and_append_to_index(fed_table_index_t *index) {
  if (index->capacity == index->num_tables) {
    fed_table_t *old_tables = index->tables;
    fed_table_t *new_tables =
        (fed_table_t *)calloc(sizeof(fed_table_t), index->capacity * 2);
    memcpy(index->tables, old_tables, sizeof(fed_table_t) * index->capacity);

    index->capacity = index->capacity * 2;
    index->tables = new_tables;
    // Unfortunately, we cannot free the old tables, in case they are being
    // accessed.
  }

  // Now we are guaranteed to have space for another table in the index.
  fed_table_t *new_table = &index->tables[index->num_tables];
  new_table->num_entries = 0;
  new_table->entries =
      (source_loc_t *)malloc(sizeof(source_loc_t) * NUM_ELEMENTS_PER_TABLE);
  index->num_tables++;

  return new_table;
}

// Initialize the FED tables list, indexed by a value of type
// fed_type_t. This is called once, by the first unit to load.
static void initialize_fed_tables() {
    fed_tables = (fed_table_index_t *)malloc(NUM_FED_TYPES * sizeof(fed_table_index_t));
    assert(fed_tables != NULL);
    for (unsigned i = 0; i < NUM_FED_TYPES; i++) {
        fed_table_index_t *index = fed_tables + i;
        index->num_tables = 0;
        index->num_total_entries = 0;
        index->capacity = DEFAULT_NUM_POINTERS_TO_TABLES;
        index->tables = (fed_table_t *)calloc(sizeof(fed_table_t),
                                             index->capacity);

        allocate_new_table_and_append_to_index(index);
    }
    fed_tables_initialized = true;
}

static inline int is_table_full(const fed_table_t *table) {
  return table->num_entries == NUM_ELEMENTS_PER_TABLE;
}

static inline fed_table_t* get_table_for_insertion(fed_table_index_t *index) {
  return &index->tables[index->num_tables - 1];
}

static inline void add_entry_to_table(fed_table_t *table,
    const source_loc_t *fed_entry) {
  assert(!is_table_full(table));
  table->entries[table->num_entries] = *fed_entry;
  table->num_entries++;
}

// Add a new FED table of the given type.
static inline void add_fed_table(fed_type_t fed_type, uint64_t num_entries,
                                 const source_loc_t *fed_entries) {
    fed_table_index_t *index = &fed_tables[fed_type];
    fed_table_t * table = get_table_for_insertion(index);
    for (uint64_t i = 0; i < num_entries; i++) {
      if (is_table_full(table)) {
        table = allocate_new_table_and_append_to_index(index);
      }
      add_entry_to_table(table, fed_entries + i);
    }

    index->num_total_entries += num_entries;
}

// The unit-local counter pointed to by 'fed_id_base' keeps track of
// that unit's "base" ID value of the given type (recall that there is
// a private ID space per FED type). The "base" ID value is the global
// ID that corresponds to the unit's local ID 0. This function stores
// the correct value into a unit's base ID.
static inline void update_ids(fed_type_t fed_type, uint64_t num_entries,
                              csi_id_t *fed_id_base) {
    fed_table_index_t *index = &fed_tables[fed_type];
    // The base ID is the current number of FED entries before adding
    // the new FED table.
    *fed_id_base = index->num_total_entries - num_entries;
}

static inline fed_table_t *get_table_for_id(const fed_table_index_t *index,
    const csi_id_t csi_id) {
    return index->tables + (csi_id / NUM_ELEMENTS_PER_TABLE);
}

static inline source_loc_t *get_entry_for_table(const fed_table_t* table,
    const csi_id_t csi_id) {
    assert(table->entries != NULL);
    return table->entries + (csi_id % NUM_ELEMENTS_PER_TABLE);
}

// Return the FED entry of the given type, corresponding to the given
// CSI ID.
static inline const source_loc_t *get_fed_entry(fed_type_t fed_type,
                                                const csi_id_t csi_id) {
    fed_table_index_t *index = &fed_tables[fed_type];
   
    if (csi_id < 0 || (uint64_t)csi_id < index->num_total_entries) {
        fed_table_t *table = get_table_for_id(index, csi_id);
        return get_entry_for_table(table, csi_id);
    } else {
        return NULL;
    }
}

// Initialize the SizeInfo tables list, indexed by a value of type
// sizeinfo_type_t. This is called once, by the first unit to load.
static void initialize_sizeinfo_tables() {
    sizeinfo_tables =
      (sizeinfo_table_t *)malloc(NUM_SIZEINFO_TYPES * sizeof(sizeinfo_table_t));
    assert(sizeinfo_tables != NULL);
    for (unsigned i = 0; i < NUM_SIZEINFO_TYPES; i++) {
        sizeinfo_table_t table;
        table.num_entries = 0;
        table.entries = NULL;
        sizeinfo_tables[i] = table;
    }
    sizeinfo_tables_initialized = true;
}

// Ensure that the SizeInfo table of the given type has enough memory
// allocated to add a new unit's entries.
static void ensure_sizeinfo_table_capacity(sizeinfo_type_t sizeinfo_type,
                                           int64_t num_new_entries) {
    if (!sizeinfo_tables_initialized) {
        initialize_sizeinfo_tables();
    }
    sizeinfo_table_t *table = &sizeinfo_tables[sizeinfo_type];
    int64_t total_num_entries = table->num_entries + num_new_entries;
    if (total_num_entries > 0) {
        table->entries = (sizeinfo_t *)realloc(table->entries,
                                               total_num_entries * sizeof(sizeinfo_t));
        table->num_entries = total_num_entries;
        assert(table->entries != NULL);
    }
}

// Add a new SizeInfo table of the given type.
static inline void add_sizeinfo_table(sizeinfo_type_t sizeinfo_type,
                                      int64_t num_entries,
                                      const sizeinfo_t *sizeinfo_entries) {
    ensure_sizeinfo_table_capacity(sizeinfo_type, num_entries);
    sizeinfo_table_t *table = &sizeinfo_tables[sizeinfo_type];
    csi_id_t base = table->num_entries - num_entries;
    for (csi_id_t i = 0; i < num_entries; i++) {
        table->entries[base + i] = sizeinfo_entries[i];
    }
}

// Return the SIZEINFO entry of the given type, corresponding to the given
// CSI ID.
static inline
const sizeinfo_t *get_sizeinfo_entry(sizeinfo_type_t sizeinfo_type,
                                     const csi_id_t csi_id) {
    // TODO(ddoucet): threadsafety
    sizeinfo_table_t *table = &sizeinfo_tables[sizeinfo_type];
    if (csi_id < table->num_entries) {
        assert(table->entries != NULL);
        return &table->entries[csi_id];
    } else {
        return NULL;
    }
}

// ------------------------------------------------------------------------
// External function definitions, including CSIRT API functions.
// ------------------------------------------------------------------------

EXTERN_C

// Not used at the moment
// __thread bool __csi_disable_instrumentation;

typedef struct {
    int64_t num_entries;
    csi_id_t *id_base;
    const source_loc_t *entries;
} unit_fed_table_t;

typedef struct {
  int64_t num_entries;
  const sizeinfo_t *entries;
} unit_sizeinfo_table_t;

// Function signature for the function (generated by the CSI compiler
// pass) that updates the callsite to function ID mappings.
typedef void (*__csi_init_callsite_to_functions)();

static inline instrumentation_counts_t compute_inst_counts(unit_fed_table_t *unit_fed_tables) {
    instrumentation_counts_t counts;
    int64_t *base = (int64_t *)&counts;
    for (unsigned i = 0; i < NUM_FED_TYPES; i++)
        *(base + i) = unit_fed_tables[i].num_entries;
    return counts;
}

_Atomic int32_t lock = 0;

// A call to this is inserted by the CSI compiler pass, and occurs
// before main().
CSIRT_API void __csirt_unit_init(
    const char * const name,
    unit_fed_table_t *unit_fed_tables,
    unit_sizeinfo_table_t *unit_sizeinfo_tables,
    __csi_init_callsite_to_functions callsite_to_func_init) {
    // Make sure we don't instrument things in __csi_init or __csi_unit init.
    // __csi_disable_instrumentation = true;

    // TODO(ddoucet): threadsafety
    if (!csi_init_called) {
        __csi_init();
        csi_init_called = true;
    }

    int32_t acquired = 0;
    while (!(acquired = atomic_compare_exchange_strong(&lock, &acquired, 1))) {}

    assert(lock == acquired == 1);

    if (!fed_tables_initialized) {
      initialize_fed_tables();
    }

    // Add all FED tables from the new unit
    for (unsigned i = 0; i < NUM_FED_TYPES; i++) {
        add_fed_table(i, unit_fed_tables[i].num_entries, unit_fed_tables[i].entries);
        update_ids(i, unit_fed_tables[i].num_entries, unit_fed_tables[i].id_base);
    }

    // Add all SizeInfo tables from the new unit
    for (int i = 0; i < NUM_SIZEINFO_TYPES; ++i) {
        add_sizeinfo_table((sizeinfo_type_t)i, unit_sizeinfo_tables[i].num_entries,
                           unit_sizeinfo_tables[i].entries);
    }

    // Initialize the callsite -> function mappings. This must happen
    // after the base IDs have been updated.
    callsite_to_func_init();

    // Call into the tool implementation.
    __csi_unit_init(name, compute_inst_counts(unit_fed_tables));

    // Reset disable flag.
    // __csi_disable_instrumentation = false;

    acquired = 1;
    assert(lock == acquired);
    int res = atomic_compare_exchange_strong(&lock, &acquired, 0);
    assert(res);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_func_source_loc(const csi_id_t func_id) {
    return get_fed_entry(FED_TYPE_FUNCTIONS, func_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_func_exit_source_loc(const csi_id_t func_exit_id) {
    return get_fed_entry(FED_TYPE_FUNCTION_EXIT, func_exit_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_bb_source_loc(const csi_id_t bb_id) {
    return get_fed_entry(FED_TYPE_BASICBLOCK, bb_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_callsite_source_loc(const csi_id_t callsite_id) {
    return get_fed_entry(FED_TYPE_CALLSITE, callsite_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_load_source_loc(const csi_id_t load_id) {
    return get_fed_entry(FED_TYPE_LOAD, load_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_store_source_loc(const csi_id_t store_id) {
    return get_fed_entry(FED_TYPE_STORE, store_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_detach_source_loc(const csi_id_t detach_id) {
  return get_fed_entry(FED_TYPE_DETACH, detach_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_task_source_loc(const csi_id_t task_id) {
  return get_fed_entry(FED_TYPE_TASK, task_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_task_exit_source_loc(
    const csi_id_t task_exit_id) {
  return get_fed_entry(FED_TYPE_TASK_EXIT, task_exit_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_detach_continue_source_loc(
    const csi_id_t detach_continue_id) {
  return get_fed_entry(FED_TYPE_DETACH_CONTINUE, detach_continue_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_sync_source_loc(const csi_id_t sync_id) {
  return get_fed_entry(FED_TYPE_SYNC, sync_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t * __csi_get_alloca_source_loc(const csi_id_t alloca_id) {
  return get_fed_entry(FED_TYPE_ALLOCA, alloca_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_allocfn_source_loc(const csi_id_t allocfn_id) {
  return get_fed_entry(FED_TYPE_ALLOCFN, allocfn_id);
}

CSIRT_API
__attribute__((const))
const source_loc_t *__csi_get_free_source_loc(const csi_id_t free_id) {
  return get_fed_entry(FED_TYPE_FREE, free_id);
}

CSIRT_API
__attribute__((const))
const sizeinfo_t *__csi_get_bb_sizeinfo(const csi_id_t bb_id) {
  return get_sizeinfo_entry(SIZEINFO_TYPE_BASICBLOCK, bb_id);
}

CSIRT_API
__attribute__((const))
const char *__csi_get_allocfn_str(const allocfn_prop_t prop) {
  return allocfn_str[prop.allocfn_ty];
}

CSIRT_API
__attribute__((const))
const char *__csi_get_free_str(const free_prop_t prop) {
  return free_str[prop.free_ty];
}

EXTERN_C_END

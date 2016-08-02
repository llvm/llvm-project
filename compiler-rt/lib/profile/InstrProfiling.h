/*===- InstrProfiling.h- Support library for PGO instrumentation ----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILE_INSTRPROFILING_H_
#define PROFILE_INSTRPROFILING_H_

#include "InstrProfilingPort.h"
#include "InstrProfData.inc"

enum ValueKind {
#define VALUE_PROF_KIND(Enumerator, Value) Enumerator = Value,
#include "InstrProfData.inc"
};

typedef void *IntPtrT;
typedef struct COMPILER_RT_ALIGNAS(INSTR_PROF_DATA_ALIGNMENT)
    __llvm_profile_data {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) Type Name;
#include "InstrProfData.inc"
} __llvm_profile_data;

typedef struct __llvm_profile_header {
#define INSTR_PROF_RAW_HEADER(Type, Name, Initializer) Type Name;
#include "InstrProfData.inc"
} __llvm_profile_header;

/*!
 * \brief Get number of bytes necessary to pad the argument to eight
 * byte boundary.
 */
uint8_t __llvm_profile_get_num_padding_bytes(uint64_t SizeInBytes);

/*!
 * \brief Get required size for profile buffer.
 */
uint64_t __llvm_profile_get_size_for_buffer(void);

/*!
 * \brief Write instrumentation data to the given buffer.
 *
 * \pre \c Buffer is the start of a buffer at least as big as \a
 * __llvm_profile_get_size_for_buffer().
 */
int __llvm_profile_write_buffer(char *Buffer);

const __llvm_profile_data *__llvm_profile_begin_data(void);
const __llvm_profile_data *__llvm_profile_end_data(void);
const char *__llvm_profile_begin_names(void);
const char *__llvm_profile_end_names(void);
uint64_t *__llvm_profile_begin_counters(void);
uint64_t *__llvm_profile_end_counters(void);

/*!
 * \brief Clear profile counters to zero.
 *
 */
void __llvm_profile_reset_counters(void);

/*!
 * \brief Counts the number of times a target value is seen.
 *
 * Records the target value for the CounterIndex if not seen before. Otherwise,
 * increments the counter associated w/ the target value.
 * void __llvm_profile_instrument_target(uint64_t TargetValue, void *Data,
 *                                       uint32_t CounterIndex);
 */
void INSTR_PROF_VALUE_PROF_FUNC(
#define VALUE_PROF_FUNC_PARAM(ArgType, ArgName, ArgLLVMType) ArgType ArgName
#include "InstrProfData.inc"
);

/*!
 * \brief Prepares the value profiling data for output.
 *
 * Returns an array of pointers to value profile data.
 */
struct ValueProfData;
struct ValueProfData **__llvm_profile_gather_value_data(uint64_t *Size);

/*!
 * \brief Write instrumentation data to the current file.
 *
 * Writes to the file with the last name given to \a *
 * __llvm_profile_set_filename(),
 * or if it hasn't been called, the \c LLVM_PROFILE_FILE environment variable,
 * or if that's not set, the last name given to
 * \a __llvm_profile_override_default_filename(), or if that's not set,
 * \c "default.profraw".
 */
int __llvm_profile_write_file(void);

/*!
 * \brief Set the filename for writing instrumentation data.
 *
 * Sets the filename to be used for subsequent calls to
 * \a __llvm_profile_write_file().
 *
 * \c Name is not copied, so it must remain valid.  Passing NULL resets the
 * filename logic to the default behaviour.
 */
void __llvm_profile_set_filename(const char *Name);

/*!
 * \brief Set the filename for writing instrumentation data, unless the
 * \c LLVM_PROFILE_FILE environment variable was set.
 *
 * Unless overridden, sets the filename to be used for subsequent calls to
 * \a __llvm_profile_write_file().
 *
 * \c Name is not copied, so it must remain valid.  Passing NULL resets the
 * filename logic to the default behaviour (unless the \c LLVM_PROFILE_FILE
 * was set in which case it has no effect).
 */
void __llvm_profile_override_default_filename(const char *Name);

/*! \brief Register to write instrumentation data to file at exit. */
int __llvm_profile_register_write_file_atexit(void);

/*! \brief Initialize file handling. */
void __llvm_profile_initialize_file(void);

/*! \brief Get the magic token for the file format. */
uint64_t __llvm_profile_get_magic(void);

/*! \brief Get the version of the file format. */
uint64_t __llvm_profile_get_version(void);

#endif /* PROFILE_INSTRPROFILING_H_ */

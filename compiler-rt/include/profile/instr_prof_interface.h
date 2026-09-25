/*===---- instr_prof_interface.h - Instrumentation PGO User Program API ----===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 *
 * This header provides a public interface for fine-grained control of counter
 * reset and profile dumping. These interface functions can be directly called
 * in user programs.
 *
\*===---------------------------------------------------------------------===*/

#ifndef COMPILER_RT_INSTR_PROFILING
#define COMPILER_RT_INSTR_PROFILING

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __LLVM_INSTR_PROFILE_GENERATE
// Profile file reset and dump interfaces.
// When `-fprofile[-instr]-generate`/`-fcs-profile-generate` is in effect,
// clang defines __LLVM_INSTR_PROFILE_GENERATE to pick up the API calls.

/*! \brief Initialize file handling. */
void __llvm_profile_initialize_file(void);

/*!
 * \brief Write instrumentation data to the current file.
 *
 * Writes to the file with the last name given to \a
 * __llvm_profile_set_filename(),
 * or if it hasn't been called, the \c LLVM_PROFILE_FILE environment variable,
 * or if that's not set, the last name set to INSTR_PROF_PROFILE_NAME_VAR,
 * or if that's not set,  \c "default.profraw".
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
 *
 * Note: There may be multiple copies of the profile runtime (one for each
 * instrumented image/DSO). This API only modifies the filename within the
 * copy of the runtime available to the calling image.
 *
 * Warning: This is a no-op if continuous mode (\ref
 * __llvm_profile_is_continuous_mode_enabled) is on. The reason for this is
 * that in continuous mode, profile counters are mmap()'d to the profile at
 * program initialization time. Support for transferring the mmap'd profile
 * counts to a new file has not been implemented.
 */
void __llvm_profile_set_filename(const char *Name);

/*!
 * \brief Return filename (including path) of the profile data. Note that if the
 * user calls __llvm_profile_set_filename later after invoking this interface,
 * the actual file name may differ from what is returned here.
 * Side-effect: this API call will invoke malloc with dynamic memory allocation
 * (the returned pointer must be passed to `free` to avoid a leak, unless
 * allocation fails, in which case a static `""` is returned).
 *
 * Note: There may be multiple copies of the profile runtime (one for each
 * instrumented image/DSO). This API only retrieves the filename from the copy
 * of the runtime available to the calling image.
 */
const char *__llvm_profile_get_filename(void);

/*!
 * \brief Interface to set all PGO counters to zero for the current process.
 *
 */
void __llvm_profile_reset_counters(void);

/*!
 * \brief this is a wrapper interface to \c __llvm_profile_write_file.
 * After this interface is invoked, an already dumped flag will be set
 * so that profile won't be dumped again during program exit.
 * Invocation of interface __llvm_profile_reset_counters will clear
 * the flag. This interface is designed to be used to collect profile
 * data from user selected hot regions. The use model is
 *      __llvm_profile_reset_counters();
 *      ... hot region 1
 *      __llvm_profile_dump();
 *      .. some other code
 *      __llvm_profile_reset_counters();
 *      ... hot region 2
 *      __llvm_profile_dump();
 *
 *  It is expected that on-line profile merging is on with \c %m specifier
 *  used in profile filename . If merging is not turned on, user is expected
 *  to invoke __llvm_profile_set_filename to specify different profile names
 *  for different regions before dumping to avoid profile write clobbering.
 */
int __llvm_profile_dump(void);

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

/*!
 * \brief Merge profile data from buffer.
 *
 * Read profile data from buffer \p Profile and merge with in-process profile
 * counters and bitmaps. The client is expected to have checked or already
 * know the profile data in the buffer matches the in-process counter
 * structure before calling it. Returns 0 (success) if the profile data is
 * valid. Upon reading invalid/corrupted profile data, returns 1 (failure).
 */
int __llvm_profile_merge_from_buffer(const char *Profile, uint64_t Size);

/*! \brief Check if profile in buffer matches the current binary.
 *
 *  Returns 0 (success) if the profile data in buffer \p Profile with size
 *  \p Size was generated by the same binary and therefore matches
 *  structurally the in-process counters and bitmaps. If the profile data in
 *  buffer is not compatible, the interface returns 1 (failure).
 */
int __llvm_profile_check_compatibility(const char *Profile, uint64_t Size);

#else

#define __llvm_profile_initialize_file()
#define __llvm_profile_write_file() (0)
#define __llvm_profile_set_filename(Name)
#define __llvm_profile_get_filename() ((const char *)0)
#define __llvm_profile_reset_counters()
#define __llvm_profile_dump() (0)
#define __llvm_profile_get_size_for_buffer() ((uint64_t)0)
#define __llvm_profile_write_buffer(Buffer) (0)
#define __llvm_profile_merge_from_buffer(Profile, Size) (0)
#define __llvm_profile_check_compatibility(Profile, Size) (0)

#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif

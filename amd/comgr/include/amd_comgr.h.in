/*******************************************************************************
*
* University of Illinois/NCSA
* Open Source License
*
* Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* with the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimers.
*
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimers in the
*       documentation and/or other materials provided with the distribution.
*
*     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
*       contributors may be used to endorse or promote products derived from
*       this Software without specific prior written permission.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
* THE SOFTWARE.
*
*******************************************************************************/

#ifndef AMD_COMGR_H_
#define AMD_COMGR_H_

#include <stddef.h>   /* size_t */
#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>  /* bool */
#endif /* __cplusplus */

/* Placeholder for calling convention and import/export macros */
#ifndef AMD_COMGR_CALL
#define AMD_COMGR_CALL
#endif

#ifndef AMD_COMGR_EXPORT_DECORATOR
#ifdef __GNUC__
#define AMD_COMGR_EXPORT_DECORATOR __attribute__ ((visibility ("default")))
#else
#define AMD_COMGR_EXPORT_DECORATOR __declspec(dllexport)
#endif
#endif

#ifndef AMD_COMGR_IMPORT_DECORATOR
#ifdef __GNUC__
#define AMD_COMGR_IMPORT_DECORATOR
#else
#define AMD_COMGR_IMPORT_DECORATOR __declspec(dllimport)
#endif
#endif

#define AMD_COMGR_API_EXPORT AMD_COMGR_EXPORT_DECORATOR AMD_COMGR_CALL
#define AMD_COMGR_API_IMPORT AMD_COMGR_IMPORT_DECORATOR AMD_COMGR_CALL

#ifndef AMD_COMGR_API
#ifdef AMD_COMGR_EXPORT
#define AMD_COMGR_API AMD_COMGR_API_EXPORT
#else
#define AMD_COMGR_API AMD_COMGR_API_IMPORT
#endif
#endif

#define AMD_COMGR_INTERFACE_VERSION_MAJOR @amd_comgr_VERSION_MAJOR@
#define AMD_COMGR_INTERFACE_VERSION_MINOR @amd_comgr_VERSION_MINOR@

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

/** \defgroup codeobjectmanager Code Object Manager
 *  @{
 *
 * @brief The code object manager is a callable library that provides
 * operations for creating and inspecting code objects.
 *
 * The library provides handles to various objects. Concurrent execution of
 * operations is supported provided all objects accessed by each concurrent
 * operation are disjoint. For example, the @p amd_comgr_data_set_t handles
 * passed to operations must be disjoint, together with all the @p
 * amd_comgr_data_t handles that have been added to it. The exception is that
 * the default device library data object handles can be non-disjoint as they
 * are imutable.
 *
 * The library supports generating and inspecting code objects that
 * contain machine code for a certain set of instruction set
 * arhitectures (isa). The set of isa supported and information about
 * the properties of the isa can be queried.
 *
 * The library supports performing an action that can take data
 * objects of one kind, and generate new data objects of another kind.
 *
 * Data objects are referenced using handles using @p
 * amd_comgr_data_t. The kinds of data objects are given
 * by @p amd_comgr_data_kind_t.
 *
 * To perform an action, two @p amd_comgr_data_set_t
 * objects are created. One is used to hold all the data objects
 * needed by an action, and other is updated by the action with all
 * the result data objects. In addition, an @p
 * amd_comgr_action_info_t is created to hold
 * information that controls the action. These are then passed to @p
 * amd_comgr_do_action to perform an action specified by
 * @p amd_comgr_action_kind_t.
 *
 * Some data objects can have associated metadata. There are
 * operations for querying this metadata.
 *
 * The default device library that satisfies the requirements of the
 * compiler action can be obtained.
 *
 * The library inspects some environment variables to aid in debugging. These
 * include:
 * - @p AMD_COMGR_SAVE_TEMPS: If this is set, and is not "0", the library does
 *   not delete temporary files generated while executing compilation actions.
 *   These files do not appear in the current working directory, but are
 *   instead left in a platform-specific temporary directory (/tmp on Linux and
 *   C:\Temp or the path found in the TEMP environment variable on Windows).
 * - @p AMD_COMGR_REDIRECT_LOGS: If this is not set, or is set to "0", logs are
 *   returned to the caller as normal. If this is set to "stdout"/"-" or
 *   "stderr", logs are instead redirected to the standard output or error
 *   stream, respectively. If this is set to any other value, it is interpreted
 *   as a filename which logs should be appended to. Logs may be redirected
 *   irrespective of whether logging is enabled.
 * - @p AMD_COMGR_EMIT_VERBOSE_LOGS: If this is set, and is not "0", logs will
 *   include additional Comgr-specific informational messages.
 */


/** \defgroup symbol_versions_group Symbol Versions
 *
 * The names used for the shared library versioned symbols.
 *
 * Every function is annotated with one of the version macros defined in this
 * section.  Each macro specifies a corresponding symbol version string.  After
 * dynamically loading the shared library with \p dlopen, the address of each
 * function can be obtained using \p dlvsym with the name of the function and
 * its corresponding symbol version string.  An error will be reported by \p
 * dlvsym if the installed library does not support the version for the
 * function specified in this version of the interface.
 *
 * @{
 */

/**
 * The function was introduced in version 1.8 of the interface and has the
 * symbol version string of ``"@amd_comgr_NAME@_1.8"``.
 */
#define AMD_COMGR_VERSION_1_8

/**
 * The function was introduced or changed in version 2.0 of the interface
 * and has the symbol version string of ``"@amd_comgr_NAME@_2.0"``.
 */
#define AMD_COMGR_VERSION_2_0

/**
 * The function was introduced or changed in version 2.2 of the interface
 * and has the symbol version string of ``"@amd_comgr_NAME@_2.2"``.
 */
#define AMD_COMGR_VERSION_2_2

/**
 * The function was introduced or changed in version 2.3 of the interface
 * and has the symbol version string of ``"@amd_comgr_NAME@_2.3"``.
 */
#define AMD_COMGR_VERSION_2_3

/**
 * The function was introduced or changed in version 2.4 of the interface
 * and has the symbol version string of ``"@amd_comgr_NAME@_2.4"``.
 */
#define AMD_COMGR_VERSION_2_4

/** @} */

/**
 * @brief Status codes.
 */
typedef enum amd_comgr_status_s {
  /**
   * The function has been executed successfully.
   */
  AMD_COMGR_STATUS_SUCCESS = 0x0,
  /**
   * A generic error has occurred.
   */
  AMD_COMGR_STATUS_ERROR = 0x1,
  /**
   * One of the actual arguments does not meet a precondition stated
   * in the documentation of the corresponding formal argument.
   */
  AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = 0x2,
  /**
   * Failed to allocate the necessary resources.
   */
  AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = 0x3,
} amd_comgr_status_t;

/**
 * @brief The source languages supported by the compiler.
 */
typedef enum amd_comgr_language_s {
  /**
   * No high level language.
   */
  AMD_COMGR_LANGUAGE_NONE = 0x0,
  /**
   * OpenCL 1.2.
   */
  AMD_COMGR_LANGUAGE_OPENCL_1_2 = 0x1,
  /**
   * OpenCL 2.0.
   */
  AMD_COMGR_LANGUAGE_OPENCL_2_0 = 0x2,
  /**
   * AMD Hetrogeneous C++ (HC).
   */
  AMD_COMGR_LANGUAGE_HC = 0x3,
  /**
   * HIP.
   */
  AMD_COMGR_LANGUAGE_HIP = 0x4,
  /**
   * Marker for last valid language.
   */
  AMD_COMGR_LANGUAGE_LAST = AMD_COMGR_LANGUAGE_HIP
} amd_comgr_language_t;

/**
 * @brief Query additional information about a status code.
 *
 * @param[in] status Status code.
 *
 * @param[out] status_string A NUL-terminated string that describes
 * the error status.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * status is an invalid status code, or @p status_string is NULL.
 */
amd_comgr_status_t AMD_COMGR_API amd_comgr_status_string(
    amd_comgr_status_t status,
    const char ** status_string) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the version of the code object manager interface
 * supported.
 *
 * An interface is backwards compatible with an implementation with an
 * equal major version, and a greater than or equal minor version.
 *
 * @param[out] major Major version number.
 *
 * @param[out] minor Minor version number.
 */
void AMD_COMGR_API amd_comgr_get_version(
  size_t *major,
  size_t *minor) AMD_COMGR_VERSION_1_8;

/**
 * @brief The kinds of data supported.
 */
typedef enum amd_comgr_data_kind_s {
  /**
   * No data is available.
   */
  AMD_COMGR_DATA_KIND_UNDEF = 0x0,
  /**
   * The data is a textual main source.
   */
  AMD_COMGR_DATA_KIND_SOURCE = 0x1,
  /**
   * The data is a textual source that is included in the main source
   * or other include source.
   */
  AMD_COMGR_DATA_KIND_INCLUDE = 0x2,
  /**
   * The data is a precompiled-header source that is included in the main
   * source or other include source.
   */
  AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER = 0x3,
  /**
   * The data is a diagnostic output.
   */
  AMD_COMGR_DATA_KIND_DIAGNOSTIC = 0x4,
  /**
   * The data is a textual log output.
   */
  AMD_COMGR_DATA_KIND_LOG = 0x5,
  /**
   * The data is compiler LLVM IR bit code for a specific isa.
   */
  AMD_COMGR_DATA_KIND_BC = 0x6,
  /**
   * The data is a relocatable machine code object for a specific isa.
   */
  AMD_COMGR_DATA_KIND_RELOCATABLE = 0x7,
  /**
   * The data is an executable machine code object for a specific
   * isa. An executable is the kind of code object that can be loaded
   * and executed.
   */
  AMD_COMGR_DATA_KIND_EXECUTABLE = 0x8,
  /**
   * The data is a block of bytes.
   */
  AMD_COMGR_DATA_KIND_BYTES = 0x9,
  /**
   * The data is a fat binary (clang-offload-bundler output).
   */
  AMD_COMGR_DATA_KIND_FATBIN = 0x10,
  /**
   * Marker for last valid data kind.
   */
  AMD_COMGR_DATA_KIND_LAST = AMD_COMGR_DATA_KIND_FATBIN
} amd_comgr_data_kind_t;

/**
 * @brief A handle to a data object.
 *
 * Data objects are used to hold the data which is either an input or
 * output of a code object manager action.
 */
typedef struct amd_comgr_data_s {
  uint64_t handle;
} amd_comgr_data_t;

/**
 * @brief A handle to an action data object.
 *
 * An action data object holds a set of data objects. These can be
 * used as inputs to an action, or produced as the result of an
 * action.
 */
typedef struct amd_comgr_data_set_s {
  uint64_t handle;
} amd_comgr_data_set_t;

/**
 * @brief A handle to an action information object.
 *
 * An action information object holds all the necessary information,
 * excluding the input data objects, required to perform an action.
 */
typedef struct amd_comgr_action_info_s {
  uint64_t handle;
} amd_comgr_action_info_t;

/**
 * @brief A handle to a metadata node.
 *
 * A metadata node handle is used to traverse the metadata associated
 * with a data node.
 */
typedef struct amd_comgr_metadata_node_s {
  uint64_t handle;
} amd_comgr_metadata_node_t;

/**
 * @brief A handle to a machine code object symbol.
 *
 * A symbol handle is used to obtain the properties of symbols of a machine code
 * object. A symbol handle is invalidated when the data object containing the
 * symbol is destroyed.
 */
typedef struct amd_comgr_symbol_s {
  uint64_t handle;
} amd_comgr_symbol_t;

/**
 * @brief A handle to a disassembly information object.
 *
 * A disassembly information object holds all the necessary information,
 * excluding the input data, required to perform disassembly.
 */
typedef struct amd_comgr_disassembly_info_s {
  uint64_t handle;
} amd_comgr_disassembly_info_t;

/**
 * @brief A handle to a symbolizer information object.
 *
 * A symbolizer information object holds all the necessary information
 * required to perform symbolization.
 */
typedef struct amd_comgr_symbolizer_info_s {
  uint64_t handle;
} amd_comgr_symbolizer_info_t;

/**
 * @brief Return the number of isa names supported by this version of
 * the code object manager library.
 *
 * The isa name specifies the instruction set architecture that should
 * be used in the actions that involve machine code generation or
 * inspection.
 *
 * @param[out] count The number of isa names supported.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * count is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_isa_count(
  size_t *count) AMD_COMGR_VERSION_2_0;

/**
 * @brief Return the Nth isa name supported by this version of the
 * code object manager library.
 *
 * @param[in] index The index of the isa name to be returned. The
 * first isa name is index 0.
 *
 * @param[out] isa_name A null terminated string that is the isa name
 * being requested.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * index is greater than the number of isa name supported by this
 * version of the code object manager library. @p isa_name is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_isa_name(
  size_t index,
  const char **isa_name) AMD_COMGR_VERSION_2_0;

 /**
 * @brief Get a handle to the metadata of an isa name.
 *
 * The structure of the returned metadata is isa name specific and versioned
 * with details specified in README.md. It can include information about the
 * limits for resources such as registers and memory addressing.
 *
 * @param[in] isa_name The isa name to query.
 *
 * @param[out] metadata A handle to the metadata of the isa name. If
 * the isa name has no metadata then the returned handle has a kind of
 * @p AMD_COMGR_METADATA_KIND_NULL. The handle must be destroyed
 * using @c amd_comgr_destroy_metadata.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * name is NULL or is not an isa name supported by this version of the
 * code object manager library. @p metadata is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_isa_metadata(
  const char *isa_name,
  amd_comgr_metadata_node_t *metadata) AMD_COMGR_VERSION_2_0;

/**
 * @brief Create a data object that can hold data of a specified kind.
 *
 * Data objects are reference counted and are destroyed when the
 * reference count reaches 0. When a data object is created its
 * reference count is 1, it has 0 bytes of data, it has an empty name,
 * and it has no metadata.
 *
 * @param[in] kind The kind of data the object is intended to hold.
 *
 * @param[out] data A handle to the data object created. Its reference
 * count is set to 1.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * kind is an invalid data kind, or @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p data is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to create the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_create_data(
  amd_comgr_data_kind_t kind,
  amd_comgr_data_t *data) AMD_COMGR_VERSION_1_8;

 /**
 * @brief Indicate that no longer using a data object handle.
 *
 * The reference count of the associated data object is
 * decremented. If it reaches 0 it is destroyed.
 *
 * @param[in] data The data object to release.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_release_data(
  amd_comgr_data_t data) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the kind of the data object.
 *
 * @param[in] data The data object to query.
 *
 * @param[out] kind The kind of data the object.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object. @p kind is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to create the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_data_kind(
  amd_comgr_data_t data,
  amd_comgr_data_kind_t *kind) AMD_COMGR_VERSION_1_8;

/**
 * @brief Set the data content of a data object to the specified
 * bytes.
 *
 * Any previous value of the data object is overwritten. Any metadata
 * associated with the data object is also replaced which invalidates
 * all metadata handles to the old metadata.
 *
 * @param[in] data The data object to update.
 *
 * @param[in] size The number of bytes in the data specified by @p bytes.
 *
 * @param[in] bytes The bytes to set the data object to. The bytes are
 * copied into the data object and can be freed after the call.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_set_data(
  amd_comgr_data_t data,
  size_t size,
  const char* bytes) AMD_COMGR_VERSION_1_8;

/**
 * @brief For the given open posix file descriptor, map a slice of the
 * file into the data object. The slice is specified by @p offset and @p size.
 * Internally this API calls amd_comgr_set_data and resets data object's
 * current state.
 *
 * @param[in, out] data The data object to update.
 *
 * @param[in] file_descriptor The native file descriptor for an open file.
 * The @p file_descriptor must not be passed into a system I/O function
 * by any other thread while this function is executing.  The offset in
 * the file descriptor may be updated based on the requested size and
 * underlying platform. The @p file_descriptor may be closed immediately
 * after this function returns.
 *
 * @param[in] offset position relative to the start of the file
 * specifying the beginning of the slice in @p file_descriptor.
 *
 * @param[in] size Size in bytes of the slice.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The operation is successful.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data is an invalid or
 * the map operation failed.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_set_data_from_file_slice(
    amd_comgr_data_t data,
    int file_descriptor,
    uint64_t offset,
    uint64_t size) AMD_COMGR_VERSION_2_3;

/**
 * @brief Set the name associated with a data object.
 *
 * When compiling, the fle name of an include directive is used to
 * reference the contents of the include data object with the same
 * name. The name may also be used for other data objects in log and
 * diagnostic output.
 *
 * @param[in] data The data object to update.
 *
 * @param[in] name A null terminated string that specifies the name to
 * use for the data object. If NULL then the name is set to the empty
 * string.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_set_data_name(
  amd_comgr_data_t data,
  const char* name) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the data contents, and/or the size of the data
 * associated with a data object.
 *
 * @param[in] data The data object to query.
 *
 * @param[in, out] size On entry, the size of @p bytes. On return, if @p bytes
 * is NULL, set to the size of the data object contents.
 *
 * @param[out] bytes If not NULL, then the first @p size bytes of the
 * data object contents is copied. If NULL, no data is copied, and
 * only @p size is updated (useful in order to find the size of buffer
 * required to copy the data).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_data(
  amd_comgr_data_t data,
  size_t *size,
  char *bytes) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the data object name and/or name length.
 *
 * @param[in] data The data object to query.
 *
 * @param[in, out] size On entry, the size of @p name. On return, the size of
 * the data object name including the terminating null character.
 *
 * @param[out] name If not NULL, then the first @p size characters of the
 * data object name are copied. If @p name is NULL, only @p size is updated
 * (useful in order to find the size of buffer required to copy the name).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_data_name(
  amd_comgr_data_t data,
  size_t *size,
  char *name) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the data object isa name and/or isa name length.
 *
 * @param[in] data The data object to query.
 *
 * @param[in, out] size On entry, the size of @p isa_name. On return, if @p
 * isa_name is NULL, set to the size of the isa name including the terminating
 * null character.
 *
 * @param[out] isa_name If not NULL, then the first @p size characters
 * of the isa name are copied. If NULL, no isa name is copied, and
 * only @p size is updated (useful in order to find the size of buffer
 * required to copy the isa name).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF, or is not an isa specific
 * kind. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_data_isa_name(
  amd_comgr_data_t data,
  size_t *size,
  char *isa_name) AMD_COMGR_VERSION_2_0;

/**
 * @brief Create a symbolizer info object.
 *
 * @param[in] code_object A data object denoting a code object for which
 * symbolization should be performed. The kind of this object must be
 * ::AMD_COMGR_DATA_KIND_RELOCATABLE, ::AMD_COMGR_DATA_KIND_EXECUTABLE,
 * or ::AMD_COMGR_DATA_KIND_BYTES.
 *
 * @param[in] print_symbol_callback Function called by a successfull
 * symbolize query. @p symbol is a null-terminated string containing the
 * symbolization of the address and @p user_data is an arbitary user data.
 * The callback does not own @p symbol, and it cannot be referenced once
 * the callback returns.
 *
 * @param[out] symbolizer_info A handle to the symbolizer info object created.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if @p code_object is
 * invalid or @p print_symbol_callback is null.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to create @p symbolizer_info as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_create_symbolizer_info(
    amd_comgr_data_t code_object,
    void (*print_symbol_callback)(
      const char *symbol,
      void *user_data),
    amd_comgr_symbolizer_info_t *symbolizer_info) AMD_COMGR_VERSION_2_4;

/**
 * @brief Destroy symbolizer info object.
 *
 * @param[in] symbolizer_info A handle to symbolizer info object to destroy.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS on successful execution.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if @p
 * symbolizer_info is invalid.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_destroy_symbolizer_info(
    amd_comgr_symbolizer_info_t symbolizer_info) AMD_COMGR_VERSION_2_4;

/**
 * @brief Symbolize an address.
 *
 * The @p address is symbolized using the symbol definitions of the
 * @p code_object specified when the @p symbolizer_info was created.
 * The @p print_symbol_callback callback function specified when the
 * @p symbolizer_info was created is called passing the
 * symbolization result as @p symbol and @p user_data value.
 *
 * If symbolization is not possible ::AMD_COMGR_STATUS_SUCCESS is returned and
 * the string passed to the @p symbol argument of the @p print_symbol_callback
 * specified when the @p symbolizer_info was created contains the text
 * "<invalid>" or "??". This is consistent with `llvm-symbolizer` utility.
 *
 * @param[in] symbolizer_info A handle to symbolizer info object which should be
 * used to symbolize the @p address.
 *
 * @param[in] address An unrelocated ELF address to which symbolization
 * query should be performed.
 *
 * @param[in] is_code if true, the symbolizer symbolize the address as code
 * and the symbolization result contains filename, function name, line number
 * and column number, else the symbolizer symbolize the address as data and
 * the symbolizaion result contains symbol name, symbol's starting address
 * and symbol size.
 *
 * @param[in] user_data Arbitrary user-data passed to @p print_symbol_callback
 * callback as described for @p symbolizer_info argument.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * symbolizer_info is an invalid data object.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_symbolize(
    amd_comgr_symbolizer_info_t symbolizer_info,
    uint64_t address,
    bool is_code,
    void *user_data) AMD_COMGR_VERSION_2_4;

 /**
 * @brief Get a handle to the metadata of a data object.
 *
 * @param[in] data The data object to query.
 *
 * @param[out] metadata A handle to the metadata of the data
 * object. If the data object has no metadata then the returned handle
 * has a kind of @p AMD_COMGR_METADATA_KIND_NULL. The
 * handle must be destroyed using @c amd_comgr_destroy_metadata.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * data is an invalid data object, or has kind @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p metadata is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_data_metadata(
  amd_comgr_data_t data,
  amd_comgr_metadata_node_t *metadata) AMD_COMGR_VERSION_1_8;

/**
 * @brief Destroy a metadata handle.
 *
 * @param[in] metadata A metadata handle to destroy.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p metadata is an invalid
 * metadata handle.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update metadata
 * handle as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_destroy_metadata(amd_comgr_metadata_node_t metadata) AMD_COMGR_VERSION_1_8;

/**
 * @brief Create a data set object.
 *
 * @param[out] data_set A handle to the data set created. Initially it
 * contains no data objects.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data_set is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to create the data
 * set object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_create_data_set(
  amd_comgr_data_set_t *data_set) AMD_COMGR_VERSION_1_8;

/**
 * @brief Destroy a data set object.
 *
 * The reference counts of any associated data objects are decremented. Any
 * handles to the data set object become invalid.
 *
 * @param[in] data_set A handle to the data set object to destroy.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data_set is an invalid
 * data set object.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update data set
 * object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_destroy_data_set(
  amd_comgr_data_set_t data_set) AMD_COMGR_VERSION_1_8;

/**
 * @brief Add a data object to a data set object if it is not already added.
 *
 * The reference count of the data object is incremented.
 *
 * @param[in] data_set A handle to the data set object to be updated.
 *
 * @param[in] data A handle to the data object to be added. If @p data_set
 * already has the specified handle present, then it is not added. The order
 * that data objects are added is preserved.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data_set is an invalid
 * data set object. @p data is an invalid data object; has undef kind; has
 * include kind but does not have a name.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update data set
 * object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_data_set_add(
  amd_comgr_data_set_t data_set,
  amd_comgr_data_t data) AMD_COMGR_VERSION_1_8;

/**
 * @brief Remove all data objects of a specified kind from a data set object.
 *
 * The reference count of the removed data objects is decremented.
 *
 * @param[in] data_set A handle to the data set object to be updated.
 *
 * @param[in] data_kind The data kind of the data objects to be removed. If @p
 * AMD_COMGR_DATA_KIND_UNDEF is specified then all data objects are removed.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data_set is an invalid
 * data set object. @p data_kind is an invalid data kind.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update data set
 * object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_data_set_remove(
  amd_comgr_data_set_t data_set,
  amd_comgr_data_kind_t data_kind) AMD_COMGR_VERSION_1_8;

/**
 * @brief Return the number of data objects of a specified data kind that are
 * added to a data set object.
 *
 * @param[in] data_set A handle to the data set object to be queried.
 *
 * @param[in] data_kind The data kind of the data objects to be counted.
 *
 * @param[out] count The number of data objects of data kind @p data_kind.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data_set is an invalid
 * data set object. @p data_kind is an invalid data kind or @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p count is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query data set
 * object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_data_count(
  amd_comgr_data_set_t data_set,
  amd_comgr_data_kind_t data_kind,
  size_t *count) AMD_COMGR_VERSION_1_8;

/**
 * @brief Return the Nth data object of a specified data kind that is added to a
 * data set object.
 *
 * The reference count of the returned data object is incremented.
 *
 * @param[in] data_set A handle to the data set object to be queried.
 *
 * @param[in] data_kind The data kind of the data object to be returned.
 *
 * @param[in] index The index of the data object of data kind @data_kind to be
 * returned. The first data object is index 0. The order of data objects matches
 * the order that they were added to the data set object.
 *
 * @param[out] data The data object being requested.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data_set is an invalid
 * data set object. @p data_kind is an invalid data kind or @p
 * AMD_COMGR_DATA_KIND_UNDEF. @p index is greater than the number of data
 * objects of kind @p data_kind. @p data is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query data set
 * object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_data_get_data(
  amd_comgr_data_set_t data_set,
  amd_comgr_data_kind_t data_kind,
  size_t index,
  amd_comgr_data_t *data) AMD_COMGR_VERSION_1_8;

/**
 * @brief Create an action info object.
 *
 * @param[out] action_info A handle to the action info object created.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to create the action info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_create_action_info(
  amd_comgr_action_info_t *action_info) AMD_COMGR_VERSION_1_8;

/**
 * @brief Destroy an action info object.
 *
 * @param[in] action_info A handle to the action info object to destroy.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_destroy_action_info(
  amd_comgr_action_info_t action_info) AMD_COMGR_VERSION_1_8;

/**
 * @brief Set the isa name of an action info object.
 *
 * When an action info object is created it has no isa name. Some
 * actions require that the action info object has an isa name
 * defined.
 *
 * @param[in] action_info A handle to the action info object to be
 * updated.
 *
 * @param[in] isa_name A null terminated string that is the isa name. If NULL
 * or the empty string then the isa name is cleared. The isa name is defined as
 * the Code Object Target Identification string, described at
 * https://llvm.org/docs/AMDGPUUsage.html#code-object-target-identification
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object. @p isa_name is not an
 * isa name supported by this version of the code object manager
 * library.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_set_isa_name(
  amd_comgr_action_info_t action_info,
  const char *isa_name) AMD_COMGR_VERSION_2_0;

/**
 * @brief Get the isa name and/or isa name length.
 *
 * @param[in] action_info The action info object to query.
 *
 * @param[in, out] size On entry, the size of @p isa_name. On return, if @p
 * isa_name is NULL, set to the size of the isa name including the terminating
 * null character.
 *
 * @param[out] isa_name If not NULL, then the first @p size characters of the
 * isa name are copied into @p isa_name. If the isa name is not set then an
 * empty string is copied into @p isa_name. If NULL, no name is copied, and
 * only @p size is updated (useful in order to find the size of buffer required
 * to copy the name).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_get_isa_name(
  amd_comgr_action_info_t action_info,
  size_t *size,
  char *isa_name) AMD_COMGR_VERSION_2_0;

/**
 * @brief Set the source language of an action info object.
 *
 * When an action info object is created it has no language defined
 * which is represented by @p
 * AMD_COMGR_LANGUAGE_NONE. Some actions require that
 * the action info object has a source language defined.
 *
 * @param[in] action_info A handle to the action info object to be
 * updated.
 *
 * @param[in] language The language to set. If @p
 * AMD_COMGR_LANGUAGE_NONE then the language is cleared.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object. @p language is an
 * invalid language.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_set_language(
  amd_comgr_action_info_t action_info,
  amd_comgr_language_t language) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the language for an action info object.
 *
 * @param[in] action_info The action info object to query.
 *
 * @param[out] language The language of the action info opject. @p
 * AMD_COMGR_LANGUAGE_NONE if not defined,
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object. @p language is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_get_language(
  amd_comgr_action_info_t action_info,
  amd_comgr_language_t *language) AMD_COMGR_VERSION_1_8;

/**
 * @brief Set the options string of an action info object.
 *
 * When an action info object is created it has an empty options string.
 *
 * This overrides any option strings or arrays previously set by calls to this
 * function or @p amd_comgr_action_info_set_option_list.
 *
 * An @p action_info object which had its options set with this function can
 * only have its option inspected with @p amd_comgr_action_info_get_options.
 *
 * @param[in] action_info A handle to the action info object to be
 * updated.
 *
 * @param[in] options A null terminated string that is the options. If
 * NULL or the empty string then the options are cleared.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action info object as out of resources.
 *
 * @deprecated since 1.3
 * @see amd_comgr_action_info_set_option_list
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_set_options(
  amd_comgr_action_info_t action_info,
  const char *options) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the options string and/or options strings length of an action
 * info object.
 *
 * The @p action_info object must have had its options set with @p
 * amd_comgr_action_info_set_options.
 *
 * @param[in] action_info The action info object to query.
 *
 * @param[in, out] size On entry, the size of @p options. On return, if @p
 * options is NULL, set to the size of the options including the terminating
 * null character.
 *
 * @param[out] options If not NULL, then the first @p size characters of
 * the options are copied. If the options are not set then an empty
 * string is copied. If NULL, options is not copied, and only @p size
 * is updated (useful inorder to find the size of buffer required to
 * copy the options).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The options of @p action_info were not set
 * with @p amd_comgr_action_info_set_options.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 *
 * @deprecated since 1.3
 * @see amd_comgr_action_info_get_option_list_count and
 * amd_comgr_action_info_get_option_list_item
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_get_options(
  amd_comgr_action_info_t action_info,
  size_t *size,
  char *options) AMD_COMGR_VERSION_1_8;

/**
 * @brief Set the options array of an action info object.
 *
 * This overrides any option strings or arrays previously set by calls to this
 * function or @p amd_comgr_action_info_set_options.
 *
 * An @p action_info object which had its options set with this function can
 * only have its option inspected with @p
 * amd_comgr_action_info_get_option_list_count and @p
 * amd_comgr_action_info_get_option_list_item.
 *
 * @param[in] action_info A handle to the action info object to be updated.
 *
 * @param[in] options An array of null terminated strings. May be NULL if @p
 * count is zero, which will result in an empty options array.
 *
 * @param[in] count The number of null terminated strings in @p options.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p action_info is an
 * invalid action info object, or @p options is NULL and @p count is non-zero.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to update action
 * info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_set_option_list(
  amd_comgr_action_info_t action_info,
  const char *options[],
  size_t count) AMD_COMGR_VERSION_1_8;

/**
 * @brief Return the number of options in the options array.
 *
 * The @p action_info object must have had its options set with @p
 * amd_comgr_action_info_set_option_list.
 *
 * @param[in] action_info The action info object to query.
 *
 * @param[out] count The number of options in the options array.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The options of @p action_info were never
 * set, or not set with @p amd_comgr_action_info_set_option_list.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p action_info is an
 * invalid action info object, or @p count is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query the data
 * object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_get_option_list_count(
  amd_comgr_action_info_t action_info,
  size_t *count) AMD_COMGR_VERSION_1_8;

/**
 * @brief Return the Nth option string in the options array and/or that
 * option's length.
 *
 * The @p action_info object must have had its options set with @p
 * amd_comgr_action_info_set_option_list.
 *
 * @param[in] action_info The action info object to query.
 *
 * @param[in] index The index of the option to be returned. The first option
 * index is 0. The order is the same as the options when they were added in @p
 * amd_comgr_action_info_set_options.
 *
 * @param[in, out] size On entry, the size of @p option. On return, if @option
 * is NULL, set to the size of the Nth option string including the terminating
 * null character.
 *
 * @param[out] option If not NULL, then the first @p size characters of the Nth
 * option string are copied into @p option. If NULL, no option string is
 * copied, and only @p size is updated (useful in order to find the size of
 * buffer required to copy the option string).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The options of @p action_info were never
 * set, or not set with @p amd_comgr_action_info_set_option_list.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p action_info is an
 * invalid action info object, @p index is invalid, or @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to query the data
 * object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_get_option_list_item(
  amd_comgr_action_info_t action_info,
  size_t index,
  size_t *size,
  char *option) AMD_COMGR_VERSION_1_8;

/**
 * @brief Set the working directory of an action info object.
 *
 * When an action info object is created it has an empty working
 * directory. Some actions use the working directory to resolve
 * relative file paths.
 *
 * @param[in] action_info A handle to the action info object to be
 * updated.
 *
 * @param[in] path A null terminated string that is the working
 * directory path. If NULL or the empty string then the working
 * directory is cleared.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_set_working_directory_path(
  amd_comgr_action_info_t action_info,
  const char *path) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the working directory path and/or working directory path
 * length of an action info object.
 *
 * @param[in] action_info The action info object to query.
 *
 * @param[in, out] size On entry, the size of @p path. On return, if @p path is
 * NULL, set to the size of the working directory path including the
 * terminating null character.
 *
 * @param[out] path If not NULL, then the first @p size characters of
 * the working directory path is copied. If the working directory path
 * is not set then an empty string is copied. If NULL, the working
 * directory path is not copied, and only @p size is updated (useful
 * in order to find the size of buffer required to copy the working
 * directory path).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_get_working_directory_path(
  amd_comgr_action_info_t action_info,
  size_t *size,
  char *path) AMD_COMGR_VERSION_1_8;

/**
 * @brief Set whether logging is enabled for an action info object.
 *
 * @param[in] action_info A handle to the action info object to be
 * updated.
 *
 * @param[in] logging Whether logging should be enabled or disable.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_set_logging(
  amd_comgr_action_info_t action_info,
  bool logging) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get whether logging is enabled for an action info object.
 *
 * @param[in] action_info The action info object to query.
 *
 * @param[out] logging Whether logging is enabled.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * action_info is an invalid action info object. @p logging is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_action_info_get_logging(
  amd_comgr_action_info_t action_info,
  bool *logging) AMD_COMGR_VERSION_1_8;

/**
 * @brief The kinds of actions that can be performed.
 */
typedef enum amd_comgr_action_kind_s {
  /**
   * Preprocess each source data object in @p input in order. For each
   * successful preprocessor invocation, add a source data object to @p result.
   * Resolve any include source names using the names of include data objects
   * in @p input. Resolve any include relative path names using the working
   * directory path in @p info. Preprocess the source for the language in @p
   * info.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any preprocessing fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name or language is not set in @p info.
   */
  AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR = 0x0,
  /**
   * Copy all existing data objects in @p input to @p output, then add the
   * device-specific and language-specific precompiled headers required for
   * compilation.
   *
   * Currently the only supported languages are @p AMD_COMGR_LANGUAGE_OPENCL_1_2
   * and @p AMD_COMGR_LANGUAGE_OPENCL_2_0.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if isa name or language
   * is not set in @p info, or the language is not supported.
   */
  AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS = 0x1,
  /**
   * Compile each source data object in @p input in order. For each
   * successful compilation add a bc data object to @p result. Resolve
   * any include source names using the names of include data objects
   * in @p input. Resolve any include relative path names using the
   * working directory path in @p info. Produce bc for isa name in @p
   * info. Compile the source for the language in @p info.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any compilation
   * fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name or language is not set in @p info.
   */
  AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = 0x2,
  /**
   * Copy all existing data objects in @p input to @p output, then add the
   * device-specific and language-specific bitcode libraries required for
   * compilation.
   *
   * Currently the only supported languages are @p AMD_COMGR_LANGUAGE_OPENCL_1_2,
   * @p AMD_COMGR_LANGUAGE_OPENCL_2_0, and @p AMD_COMGR_LANGUAGE_HIP.
   *
   * The options in @p info should be set to a set of language-specific flags.
   * For OpenCL and HIP these include:
   *
   *    correctly_rounded_sqrt
   *    daz_opt
   *    finite_only
   *    unsafe_math
   *    wavefrontsize64
   *
   * For example, to enable daz_opt and unsafe_math, the options should be set
   * as:
   *
   *    const char *options[] = {"daz_opt, "unsafe_math"};
   *    size_t optionsCount = sizeof(options) / sizeof(options[0]);
   *    amd_comgr_action_info_set_option_list(info, options, optionsCount);
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if isa name or language
   * is not set in @p info, the language is not supported, an unknown
   * language-specific flag is supplied, or a language-specific flag is
   * repeated.
   *
   * @deprecated since 1.7
   * @warning This action, followed by @c AMD_COMGR_ACTION_LINK_BC_TO_BC, may
   * result in subtle bugs due to incorrect linking of the device libraries.
   * The @c AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC action can
   * be used as a workaround which ensures the link occurs correctly.
   */
  AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES = 0x3,
  /**
   * Link each bc data object in @p input together and add the linked
   * bc data object to @p result. Any device library bc data object
   * must be explicitly added to @p input if needed.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if the link fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info and does not match the isa name
   * of all bc data objects in @p input.
   */
  AMD_COMGR_ACTION_LINK_BC_TO_BC = 0x4,
  /**
   * Optimize each bc data object in @p input and create an optimized bc data
   * object to @p result.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if the optimization fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info and does not match the isa name
   * of all bc data objects in @p input.
   */
  AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC = 0x5,
  /**
   * Perform code generation for each bc data object in @p input in
   * order. For each successful code generation add a relocatable data
   * object to @p result.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any code
   * generation fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info and does not match the isa name
   * of all bc data objects in @p input.
   */
  AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 0x6,
  /**
   * Perform code generation for each bc data object in @p input in
   * order. For each successful code generation add an assembly source data
   * object to @p result.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any code
   * generation fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info and does not match the isa name
   * of all bc data objects in @p input.
   */
  AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY = 0x7,
  /**
   * Link each relocatable data object in @p input together and add
   * the linked relocatable data object to @p result. Any device
   * library relocatable data object must be explicitly added to @p
   * input if needed.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if the link fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info and does not match the isa name
   * of all relocatable data objects in @p input.
   */
  AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE = 0x8,
  /**
   * Link each relocatable data object in @p input together and add
   * the linked executable data object to @p result. Any device
   * library relocatable data object must be explicitly added to @p
   * input if needed.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if the link fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info and does not match the isa name
   * of all relocatable data objects in @p input.
   */
  AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 0x9,
  /**
   * Assemble each source data object in @p input in order into machine code.
   * For each successful assembly add a relocatable data object to @p result.
   * Resolve any include source names using the names of include data objects in
   * @p input. Resolve any include relative path names using the working
   * directory path in @p info. Produce relocatable for isa name in @p info.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any assembly fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if isa name is not set in
   * @p info.
   */
  AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE = 0xA,
  /**
   * Disassemble each relocatable data object in @p input in
   * order. For each successful disassembly add a source data object to
   * @p result.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any disassembly
   * fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info and does not match the isa name
   * of all relocatable data objects in @p input.
   */
  AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE = 0xB,
  /**
   * Disassemble each executable data object in @p input in order. For
   * each successful disassembly add a source data object to @p result.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any disassembly
   * fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info and does not match the isa name
   * of all relocatable data objects in @p input.
   */
  AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE = 0xC,
  /**
   * Disassemble each bytes data object in @p input in order. For each
   * successful disassembly add a source data object to @p
   * result. Only simple assembly language commands are generate that
   * corresponf to raw bytes are supported, not any directives that
   * control the code object layout, or symbolic branch targets or
   * names.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any disassembly
   * fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name is not set in @p info
   */
  AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE = 0xD,
  /**
   * Compile each source data object in @p input in order. For each
   * successful compilation add a fat binary to @p result. Resolve
   * any include source names using the names of include data objects
   * in @p input. Resolve any include relative path names using the
   * working directory path in @p info. Produce fat binary for isa name in @p
   * info. Compile the source for the language in @p info.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any compilation
   * fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name or language is not set in @p info.
   */
  AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN = 0xE,
  /**
   * Compile each source data object in @p input in order. For each
   * successful compilation add a bc data object to @p result. Resolve
   * any include source names using the names of include data objects
   * in @p input. Resolve any include relative path names using the
   * working directory path in @p info. Produce bc for isa name in @p
   * info. Compile the source for the language in @p info. Link against
   * the device-specific and language-specific bitcode device libraries
   * required for compilation.
   *
   * Return @p AMD_COMGR_STATUS_ERROR if any compilation
   * fails.
   *
   * Return @p AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT
   * if isa name or language is not set in @p info.
   */
  AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 0xF,
  /**
   * Marker for last valid action kind.
   */
  AMD_COMGR_ACTION_LAST = AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC
} amd_comgr_action_kind_t;

/**
 * @brief Perform an action.
 *
 * Each action ignores any data objects in @p input that it does not
 * use. If logging is enabled in @info then @p result will have a log
 * data object added. Any diagnostic data objects produced by the
 * action will be added to @p result. See the description of each
 * action in @p amd_comgr_action_kind_t.
 *
 * @param[in] kind The action to perform.
 *
 * @param[in] info The action info to use when performing the action.
 *
 * @param[in] input The input data objects to the @p kind action.
 *
 * @param[out] result Any data objects are removed before performing
 * the action which then adds all data objects produced by the action.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR An error was
 * reported when executing the action.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * kind is an invalid action kind. @p input_data or @p result_data are
 * invalid action data object handles. See the description of each
 * action in @p amd_comgr_action_kind_t for other
 * conditions that result in this status.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_do_action(
  amd_comgr_action_kind_t kind,
  amd_comgr_action_info_t info,
  amd_comgr_data_set_t input,
  amd_comgr_data_set_t result) AMD_COMGR_VERSION_1_8;

/**
 * @brief The kinds of metadata nodes.
 */
typedef enum amd_comgr_metadata_kind_s {
  /**
   * The NULL metadata handle.
   */
  AMD_COMGR_METADATA_KIND_NULL = 0x0,
  /**
   * A sting value.
   */
  AMD_COMGR_METADATA_KIND_STRING = 0x1,
  /**
   * A map that consists of a set of key and value pairs.
   */
  AMD_COMGR_METADATA_KIND_MAP = 0x2,
  /**
   * A list that consists of a sequence of values.
   */
  AMD_COMGR_METADATA_KIND_LIST = 0x3,
  /**
   * Marker for last valid metadata kind.
   */
  AMD_COMGR_METADATA_KIND_LAST = AMD_COMGR_METADATA_KIND_LIST
} amd_comgr_metadata_kind_t;

/**
 * @brief Get the kind of the metadata node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[out] kind The kind of the metadata node.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node. @p kind is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to create the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_metadata_kind(
  amd_comgr_metadata_node_t metadata,
  amd_comgr_metadata_kind_t *kind) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the string and/or string length from a metadata string
 * node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[in, out] size On entry, the size of @p string. On return, if @p
 * string is NULL, set to the size of the string including the terminating null
 * character.
 *
 * @param[out] string If not NULL, then the first @p size characters
 * of the string are copied. If NULL, no string is copied, and only @p
 * size is updated (useful in order to find the size of buffer required
 * to copy the string).
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or does not have kind @p
 * AMD_COMGR_METADATA_KIND_STRING. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_metadata_string(
  amd_comgr_metadata_node_t metadata,
  size_t *size,
  char *string) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the map size from a metadata map node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[out] size The number of entries in the map.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or not of kind @p
 * AMD_COMGR_METADATA_KIND_MAP. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_metadata_map_size(
  amd_comgr_metadata_node_t metadata,
  size_t *size) AMD_COMGR_VERSION_1_8;

/**
 * @brief Iterate over the elements a metadata map node.
 *
 * @warning The metadata nodes which are passed to the callback are not owned
 * by the callback, and are freed just after the callback returns. The callback
 * must not save any references to its parameters between iterations.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[in] callback The function to call for each entry in the map. The
 * entry's key is passed in @p key, the entry's value is passed in @p value, and
 * @p user_data is passed as @p user_data. If the function returns with a status
 * other than @p AMD_COMGR_STATUS_SUCCESS then iteration is stopped.
 *
 * @param[in] user_data The value to pass to each invocation of @p
 * callback. Allows context to be passed into the call back function.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR An error was
 * reported by @p callback.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or not of kind @p
 * AMD_COMGR_METADATA_KIND_MAP. @p callback is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to iterate the metadata as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_iterate_map_metadata(
  amd_comgr_metadata_node_t metadata,
  amd_comgr_status_t (*callback)(
    amd_comgr_metadata_node_t key,
    amd_comgr_metadata_node_t value,
    void *user_data),
  void *user_data) AMD_COMGR_VERSION_1_8;

/**
 * @brief Use a string key to lookup an element of a metadata map
 * node and return the entry value.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[in] key A null terminated string that is the key to lookup.
 *
 * @param[out] value The metadata node of the @p key element of the
 * @p metadata map metadata node. The handle must be destroyed
 * using @c amd_comgr_destroy_metadata.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The map has no entry
 * with a string key with the value @p key.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or not of kind @p
 * AMD_COMGR_METADATA_KIND_MAP. @p key or @p value is
 * NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to lookup metadata as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_metadata_lookup(
  amd_comgr_metadata_node_t metadata,
  const char *key,
  amd_comgr_metadata_node_t *value) AMD_COMGR_VERSION_1_8;

/**
 * @brief Get the list size from a metadata list node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[out] size The number of entries in the list.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node, or does nopt have kind @p
 * AMD_COMGR_METADATA_KIND_LIST. @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_get_metadata_list_size(
  amd_comgr_metadata_node_t metadata,
  size_t *size) AMD_COMGR_VERSION_1_8;

/**
 * @brief Return the Nth metadata node of a list metadata node.
 *
 * @param[in] metadata The metadata node to query.
 *
 * @param[in] index The index being requested. The first list element
 * is index 0.
 *
 * @param[out] value The metadata node of the @p index element of the
 * @p metadata list metadata node. The handle must be destroyed
 * using @c amd_comgr_destroy_metadata.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p
 * metadata is an invalid metadata node or not of kind @p
 * AMD_COMGR_METADATA_INFO_LIST. @p index is greater
 * than the number of list elements. @p value is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to update action data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_index_list_metadata(
  amd_comgr_metadata_node_t metadata,
  size_t index,
  amd_comgr_metadata_node_t *value) AMD_COMGR_VERSION_1_8;

/**
 * @brief Iterate over the symbols of a machine code object.
 *
 * For a AMD_COMGR_DATA_KIND_RELOCATABLE the symbols in the ELF symtab section
 * are iterated. For a AMD_COMGR_DATA_KIND_EXECUTABLE the symbols in the ELF
 * dynsymtab are iterated.
 *
 * @param[in] data The data object to query.
 *
 * @param[in] callback The function to call for each symbol in the machine code
 * data object. The symbol handle is passed in @p symbol and @p user_data is
 * passed as @p user_data. If the function returns with a status other than @p
 * AMD_COMGR_STATUS_SUCCESS then iteration is stopped.
 *
 * @param[in] user_data The value to pass to each invocation of @p
 * callback. Allows context to be passed into the call back function.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR An error was
 * reported by @p callback.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data is an invalid data
 * object, or not of kind @p AMD_COMGR_DATA_KIND_RELOCATABLE or
 * AMD_COMGR_DATA_KIND_EXECUTABLE. @p callback is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to iterate the data object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_iterate_symbols(
  amd_comgr_data_t data,
  amd_comgr_status_t (*callback)(
    amd_comgr_symbol_t symbol,
    void *user_data),
  void *user_data) AMD_COMGR_VERSION_1_8;

/**
 * @brief Lookup a symbol in a machine code object by name.
 *
 * For a AMD_COMGR_DATA_KIND_RELOCATABLE the symbols in the ELF symtab section
 * are inspected. For a AMD_COMGR_DATA_KIND_EXECUTABLE the symbols in the ELF
 * dynsymtab are inspected.
 *
 * @param[in] data The data object to query.
 *
 * @param[in] name A null terminated string that is the symbol name to lookup.
 *
 * @param[out] symbol The symbol with the @p name.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The machine code object has no symbol
 * with @p name.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data is an invalid data
 * object, or not of kind @p AMD_COMGR_DATA_KIND_RELOCATABLE or
 * AMD_COMGR_DATA_KIND_EXECUTABLE.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to lookup symbol as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_symbol_lookup(
  amd_comgr_data_t data,
  const char *name,
  amd_comgr_symbol_t *symbol) AMD_COMGR_VERSION_1_8;

/**
 * @brief Machine code object symbol type.
 */
typedef enum amd_comgr_symbol_type_s {
  /**
   * The symbol's type is unknown.
   *
   * The user should not infer any specific type for symbols which return
   * `AMD_COMGR_SYMBOL_TYPE_UNKNOWN`, and these symbols may return different
   * types in future releases.
  */
  AMD_COMGR_SYMBOL_TYPE_UNKNOWN = -0x1,

  /**
   * The symbol's type is not specified.
  */
  AMD_COMGR_SYMBOL_TYPE_NOTYPE = 0x0,

  /**
   * The symbol is associated with a data object, such as a variable, an array,
   * and so on.
  */
  AMD_COMGR_SYMBOL_TYPE_OBJECT = 0x1,

  /**
   * The symbol is associated with a function or other executable code.
  */
  AMD_COMGR_SYMBOL_TYPE_FUNC = 0x2,

  /**
   * The symbol is associated with a section. Symbol table entries of this type
   * exist primarily for relocation.
  */
  AMD_COMGR_SYMBOL_TYPE_SECTION = 0x3,

  /**
   * Conventionally, the symbol's name gives the name of the source file
   * associated with the object file.
  */
  AMD_COMGR_SYMBOL_TYPE_FILE = 0x4,

  /**
   * The symbol labels an uninitialized common block.
  */
  AMD_COMGR_SYMBOL_TYPE_COMMON = 0x5,

  /**
   * The symbol is associated with an AMDGPU Code Object V2 kernel function.
  */
  AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL = 0xa
} amd_comgr_symbol_type_t;

/**
 * @brief Machine code object symbol attributes.
 */
typedef enum amd_comgr_symbol_info_s {
  /**
   * The length of the symbol name in bytes. Does not include the NUL
   * terminator. The type of this attribute is uint64_t.
  */
  AMD_COMGR_SYMBOL_INFO_NAME_LENGTH = 0x0,

  /**
   * The name of the symbol. The type of this attribute is character array with
   * the length equal to the value of the @p AMD_COMGR_SYMBOL_INFO_NAME_LENGTH
   * attribute plus 1 for a NUL terminator.
  */
  AMD_COMGR_SYMBOL_INFO_NAME = 0x1,

  /**
   * The kind of the symbol. The type of this attribute is @p
   * amd_comgr_symbol_type_t.
   */
  AMD_COMGR_SYMBOL_INFO_TYPE = 0x2,

  /**
   * Size of the variable. The value of this attribute is undefined if the
   * symbol is not a variable. The type of this attribute is uint64_t.
   */
  AMD_COMGR_SYMBOL_INFO_SIZE = 0x3,

  /**
   * Indicates whether the symbol is undefined. The type of this attribute is
   * bool.
   */
  AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED = 0x4,

  /**
   * The value of the symbol. The type of this attribute is uint64_t.
   */
  AMD_COMGR_SYMBOL_INFO_VALUE = 0x5,

  /**
   * Marker for last valid symbol info.
   */
  AMD_COMGR_SYMBOL_INFO_LAST = AMD_COMGR_SYMBOL_INFO_VALUE
} amd_comgr_symbol_info_t;

/**
 * @brief Query information about a machine code object symbol.
 *
 * @param[in] symbol The symbol to query.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of attribute, the behavior is undefined. The
 * type of value returned is specified by @p amd_comgr_symbol_info_t.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has
 * been executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The @p symbol does not have the requested @p
 * attribute.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p symbol is an invalid
 * symbol. @p attribute is an invalid value. @p value is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES
 * Unable to query symbol as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_symbol_get_info(
  amd_comgr_symbol_t symbol,
  amd_comgr_symbol_info_t attribute,
  void *value) AMD_COMGR_VERSION_1_8;

/**
 * @brief Create a disassembly info object.
 *
 * @param[in] isa_name A null terminated string that is the isa name of the
 * target to disassemble for. The isa name is defined as the Code Object Target
 * Identification string, described at
 * https://llvm.org/docs/AMDGPUUsage.html#code-object-target-identification
 *
 * @param[in] read_memory_callback Function called to request @p size bytes
 * from the program address space at @p from be read into @p to. The requested
 * @p size is never zero. Returns the number of bytes which could be read, with
 * the guarantee that no additional bytes will be available in any subsequent
 * call.
 *
 * @param[in] print_instruction_callback Function called after a successful
 * disassembly. @p instruction is a null terminated string containing the
 * disassembled instruction. The callback does not own @p instruction, and it
 * cannot be referenced once the callback returns.
 *
 * @param[in] print_address_annotation_callback Function called after @c
 * print_instruction_callback returns, once for each instruction operand which
 * was resolved to an absolute address. @p address is the absolute address in
 * the program address space. It is intended to append a symbolic
 * form of the address, perhaps as a comment, after the instruction disassembly
 * produced by @c print_instruction_callback.
 *
 * @param[out] disassembly_info A handle to the disassembly info object
 * created.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The disassembly info object was created.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p isa_name is NULL or
 * invalid; or @p read_memory_callback, @p print_instruction_callback,
 * or @p print_address_annotation_callback is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to create the
 * disassembly info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_create_disassembly_info(
  const char *isa_name,
  uint64_t (*read_memory_callback)(
    uint64_t from,
    char *to,
    uint64_t size,
    void *user_data),
  void (*print_instruction_callback)(
    const char *instruction,
    void *user_data),
  void (*print_address_annotation_callback)(
    uint64_t address,
    void *user_data),
  amd_comgr_disassembly_info_t *disassembly_info) AMD_COMGR_VERSION_2_0;

/**
 * @brief Destroy a disassembly info object.
 *
 * @param[in] disassembly_info A handle to the disassembly info object to
 * destroy.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The disassembly info object was
 * destroyed.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p disassembly_info is an
 * invalid disassembly info object.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to destroy the
 * disassembly info object as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_destroy_disassembly_info(
  amd_comgr_disassembly_info_t disassembly_info) AMD_COMGR_VERSION_1_8;

/**
 * @brief Disassemble a single instruction.
 *
 * @param[in] address The address of the first byte of the instruction in the
 * program address space.
 *
 * @param[in] user_data Arbitrary user-data passed to each callback function
 * during disassembly.
 *
 * @param[out] size The number of bytes consumed to decode the
 * instruction, or consumed while failing to decode an invalid instruction.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The disassembly was successful.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The disassembly failed.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p disassembly_info is
 * invalid or @p size is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Unable to disassemble the
 * instruction as out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_disassemble_instruction(
  amd_comgr_disassembly_info_t disassembly_info,
  uint64_t address,
  void *user_data,
  uint64_t *size) AMD_COMGR_VERSION_1_8;

/**
 * @brief Demangle a symbol name.
 *
 * @param[in] mangled_symbol_name A data object of kind @p
 * AMD_COMGR_DATA_KIND_BYTES containing the mangled symbol name.
 *
 * @param[out] demangled_symbol_name A handle to the data object of kind @p
 * AMD_COMGR_DATA_KIND_BYTES created and set to contain the demangled symbol
 * name in case of successful completion. The handle must be released using
 * @c amd_comgr_release_data. @p demangled_symbol_name is not updated for
 * an error case.
 *
 * @note If the @p mangled_symbol_name cannot be demangled, it will be copied
 * without changes to the @p demangled_symbol_name and AMD_COMGR_STATUS_SUCCESS
 * is returned.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function executed successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p mangled_symbol_name is
 * an invalid data object or not of kind @p AMD_COMGR_DATA_KIND_BYTES or
 * @p demangled_symbol_name is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES Out of resources.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_demangle_symbol_name(
    amd_comgr_data_t mangled_symbol_name,
    amd_comgr_data_t *demangled_symbol_name) AMD_COMGR_VERSION_2_2;

/**
 * @brief A data structure for Code object information.
 */
typedef struct code_object_info_s {
  /**
   * ISA name representing the code object.
   */
  const char *isa;
  /**
   * The size of the code object.
   */
  size_t size;
  /*
   * The location of code object from the beginning
   * of code object bundle.
   */
  uint64_t offset;
} amd_comgr_code_object_info_t;

/**
 * @ brief Given a bundled code object and list of target id strings, extract
 * correponding code object information.
 *
 * @param[in] data The data object for bundled code object. This should be
 * of kind AMD_COMGR_DATA_KIND_FATBIN or AMD_COMGR_DATA_KIND_EXECUTABLE or
 * AMD_COMGR_DATA_KIND_BYTES. The API interprets the data object of kind
 * AMD_COMGR_DATA_KIND_FATBIN as a clang offload bundle and of kind
 * AMD_COMGR_DATA_KIND_EXECUTABLE as an executable shared object. For a data
 * object of type AMD_COMGR_DATA_KIND_BYTES the API first inspects the data
 * passed to determine if it is a fatbin or an executable and performs
 * the lookup.
 *
 * @param[in, out] info_list A list of code object information structure
 * initialized with null terminated target id strings. If the target id
 * is matched in the code object bundle the corresponding code object
 * information is updated with offset and size of the code object. If the
 * target id is not found the offset and size are set to 0.
 *
 * @retval ::AMD_COMGR_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR The code object bundle header is incorrect
 * or reading bundle entries failed.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT @p data is not of
 * kind AMD_COMGR_DATA_KIND_FATBIN, or AMD_COMGR_DATA_KIND_BYTES or
 * AMD_COMGR_DATA_KIND_EXECUTABLE or either @p info_list is NULL.
 *
 * @retval ::AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT if the @p data has
 * invalid data.
 */
amd_comgr_status_t AMD_COMGR_API
amd_comgr_lookup_code_object(
    amd_comgr_data_t data,
    amd_comgr_code_object_info_t *info_list,
    size_t info_list_size) AMD_COMGR_VERSION_2_3;

/** @} */

#ifdef __cplusplus
}  /* end extern "C" block */
#endif

#endif  /* header guard */

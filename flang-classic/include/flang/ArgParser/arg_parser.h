/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief C header file for the argument parser
 *
 * Rules of the game:
 * - Argument parser exepects usual argc, argv pair; tries to handle all
 *   arguments
 * - Swithches start with a '-'
 * - First or last argument without a leading dash is considered the input file
 *   name (only one), everything else is treated as a switch followed by zero
 *   or more values
 */

#include "flang/ArgParser/debug_action.h"
#include <stdbool.h>

/** \brief Inform (verbosity) level
 */
typedef enum inform_level_ {
  LV_Inform = 1,
  LV_Warn,
  LV_Severe,
  LV_Fatal,
} inform_level_t;

/** \brief argument parser data structure
 *
 * Forward declared to limit access outside of this module
 */
typedef struct arg_parser_ arg_parser_t;

/** \brief Allocate and initialize argument parser data structure
 *
 * \param parser               Argument parser structure to initialize
 * \param fail_on_unknown_args Set to true to throw an error on unknown
 *                              arguments, to false to silently continue parsing
 */
void create_arg_parser(arg_parser_t **parser, bool fail_on_unknown_args);

/** \brief Deallocate argument parser data structure
 */
void destroy_arg_parser(arg_parser_t **parser);

/** \brief Register a string argument
 *
 * Prepare to set a string value for an arugument. Do not perform any
 * allocation, value from argv array will just be pointed to.
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param target         pointer to string buffer to write argument value to
 * \param default_value  value to intialize the target
 */
void register_string_arg(arg_parser_t *parser, const char *arg_name,
                         const char **target, const char *default_value);

/** \brief Register a string list argument
 *
 * Add a "string list argument", that would produce a null terminated list of
 * string pointers. Values are taken from argument pointers, but space for them
 * needs to be allocated and freed by the consumer (argc can be a fair guess of
 * the number of array elements to allocate).
 *
 * XXX If there would be many arguments of this type, then we would need to
 * dynamically allocate the array (otherwise it would be too vasteful for the
 * consumer to do that).
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param target         pointer to start of the string buffer array
 */
void register_string_list_arg(arg_parser_t *parser, const char *arg_name,
                              char **target);

/** \brief Register a combines bool and string argument
 *
 * The argument will set bool value to true if it is present and would also set
 * the string target to the value that follows it. Boolean target is
 * initialized to false and string target to NULL
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param bool_target    pointer to write boolean value to
 * \param string_target  pointer to string buffer to write string value to
 */
void register_combined_bool_string_arg(arg_parser_t *parser,
                                       const char *arg_name, bool *bool_target,
                                       const char **string_target);

/** \brief Register an integer argument
 *
 * Prepare to set an integer value for an arugument. Value will be written to a
 * location pointed by "target" argument.
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param target         pointer to location to write argument value to
 * \param default_value  value to intialize the target
 */
void register_integer_arg(arg_parser_t *parser, const char *arg_name,
                          int *target, const int default_value);

/** \brief Register a boolean argument
 *
 * Boolean argument is actually two argument: <arg> and no<arg>. First one sets
 * its target to true if it is present on command line, and the second one --
 * to false.
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param target         pointer to location to write argument value to
 * \param default_value  value to intialize the target
 */
void register_boolean_arg(arg_parser_t *parser, const char *arg_name,
                          bool *target, const bool default_value);

/** \brief Register a "q flag" argument
 *
 * "q" command line switch sets a mask in debug features array that is consumed
 * by other parts of compiler. This function also fills the array with zeros.
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param qflags         pointer to the beginning of q flags array
 * \param qflags_size    number of elelemts in qflags array
 */
void register_qflag_arg(arg_parser_t *parser, const char *arg_name, int *qflags,
                        const int qflags_size);

/** \brief Register X version of "x flag" argument
 *
 * "x" command line switch sets a mask in x flags (features) array that is
 * consumed by other parts of compiler.
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param xflags         pointer to x flags
 */
void register_xflag_arg(arg_parser_t *parser, const char *arg_name,
                        int *xflags);

/** \brief Register Y version of "x flag" argument
 *
 * "y" command line switch clears a mask in features array, complementing what
 * "x" flag does.
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param xflags         pointer to x flags
 * \param xflags_size    number of elelemts in xflags array
 */
void register_yflag_arg(arg_parser_t *parser, const char *arg_name,
                        int *xflags);

/** \brief Register "inform level" argument that determines how verbose compiler
 * output is
 *
 * Parameter would mactch sting input to information level and store it as an
 * integer constant. Recognized values and their integer codes are (minus
 * quotes):
 * - "inform" 1 (LV_Inform)
 * - "warn"   2 (LV_Warn)
 * - "severe" 3 (LV_Severe)
 * - "fatal"  4 (LV_Fatal)
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param target         pointer to location to write argument value to
 * \param default_value  value to intialize the target
 */
void register_inform_level_arg(arg_parser_t *parser, const char *arg_name,
                               inform_level_t *target,
                               const inform_level_t default_value);

/** \brief Regiser "action list" argument
 *
 * Argument would take input action list and pull relevant parts of it into the
 * target value as they are passed in as arguments. This argument type expects
 * two values: '+' separated list of new action names followed by '+' separated
 * list of original action names. The target list would contain the same
 * actions but under the new keywords (if original keywords are found in the
 * input action list).
 *
 * \param parser         argument parser data structure
 * \param arg_name       name of the argument (minus the '-')
 * \param target         pointer to location to write argument value to
 * \param default_value  value to intialize the target
 */
void register_action_map_arg(arg_parser_t *parser, const char *arg_name,
                             action_map_t *target, const action_map_t *input);

/** \brief Register input file
 *
 * This argument does not have a default value, parser would live target
 * unchanged if it does not detect input file name.
 *
 * \param parser argument parser data structure
 * \param target location to set the result (pointer to the string)
 */
void register_filename_arg(arg_parser_t *parser, const char **target);

/** \brief Parse arguments
 *
 * Set values to the variable(s) that were registered with this parser
 *
 * \param parser The parser data structure
 * \param argc   Standard argc parameter
 * \param argv   Standard argv parameter
 */
void parse_arguments(const arg_parser_t *parser, int argc, char **argv);

/** \brief Check if a value was set during parse
 *
 * \param parser   argument parser data structure
 * \param location pointer to argument's value (needs to match "register" call)
 * \return         true if value was set, false otherwise
 */
bool was_value_set(const arg_parser_t *parser, const void *location);

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Argument parser implementation
 */

#include "flang/ADT/hash.h"
#include "flang/ArgParser/arg_parser.h"
#include "flang/ArgParser/xflag.h"
#include "flang/Error/pgerror.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/** \brief Internal representation of argument parser */
struct arg_parser_ {
  /** Hash map from string keys to instances of value data structure */
  hashmap_t flags;
  /** Registered values */
  hashset_t values;
  /** Values set by the parser */
  hashset_t value_hits;
  /** Set to true to throw an error on unknown arguments, to false to silently
   * continue parsing */
  bool fail_on_unknown_args;
  /** Where to write input file name */
  const char **input_file_name_ptr;
};

/** \brief Link a bool * and a char ** value together */
typedef struct bool_string_ {
  bool *bool_ptr;
  const char **string_ptr;
} bool_string_t;

/** \brief Combine input and output for action map arguments */
typedef struct action_map_bundle_ {
  const action_map_t *input;
  action_map_t *output;
} action_map_bundle_t;

/** \brief Argument value type */
typedef enum value_type_ {
  ARG_ActionMap,
  ARG_Boolean,
  ARG_CombinedBoolean,
  ARG_InformLevel,
  ARG_Integer,
  ARG_ReverseBoolean,
  ARG_String,
  ARG_StringList,
  ARG_QFlag,
  ARG_XFlag,
  ARG_YFlag,
} value_type_t;

/** \brief Argument type and location */
typedef struct value_ {
  /** Type of value to write */
  value_type_t type;
  /** Where to write it */
  void *location;
} value_t;

static void add_generic_argument(arg_parser_t *parser, const char *arg_name,
                                 value_type_t value_type, void *value_ptr);
static void deallocate_arg_value(hash_key_t ignore, hash_data_t value_ptr,
                                 void *ignore_context);
static void compose_and_throw(const char *first_part, const char *second_part);
static char *next_value(char **argv, int *arg_index);
static inform_level_t strtoinform(const char *string);

/** Allocate argument parser hash map */
void
create_arg_parser(arg_parser_t **parser, bool fail_on_unknown_args)
{
  *parser = (arg_parser_t*) malloc(sizeof(arg_parser_t));
  (*parser)->flags = hashmap_alloc(hash_functions_strings);
  (*parser)->values = hashset_alloc(hash_functions_direct);
  (*parser)->value_hits = hashset_alloc(hash_functions_direct);
  (*parser)->fail_on_unknown_args = fail_on_unknown_args;
  (*parser)->input_file_name_ptr = NULL;
}

/** Deallocate argument parser hash map */
void
destroy_arg_parser(arg_parser_t **parser)
{
  /* Deallocate entries */
  hashmap_iterate((*parser)->flags, deallocate_arg_value, NULL);

  /* Free flags data structure */
  hashmap_free((*parser)->flags);

  /* Deallocate the data structure itself */
  free(*parser);
  *parser = NULL;
}

/** Deallocate hashtable entry */
static void
deallocate_arg_value(hash_key_t key, hash_data_t value_ptr, void *key_context)
{
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  value_t *p = (value_t *)value_ptr;
  char *k = (char *)key;
#pragma GCC diagnostic pop
  /* Some of the argument types require deallocation */
  switch (p->type) {
  case ARG_ActionMap:
  case ARG_CombinedBoolean:
    free(p->location);
    break;
  case ARG_ReverseBoolean:
    free(k);
    break;
  default:
    /* Do nothing */
    break;
  }
  free(p);
}

/** Register a string argument */
void
register_string_arg(arg_parser_t *parser, const char *arg_name,
                    const char **target, const char *default_value)
{
  /* Set default value */
  *target = default_value;

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
  add_generic_argument(parser, arg_name, ARG_String, (void *)target);
#pragma GCC diagnostic pop
}

/** Register a string list argument */
void
register_string_list_arg(arg_parser_t *parser, const char *arg_name,
                         char **target)
{
  /* Terminate list */
  *target = NULL;

  add_generic_argument(parser, arg_name, ARG_StringList, (void *)target);
}

/** Register an integer argument */
void
register_integer_arg(arg_parser_t *parser, const char *arg_name, int *target,
                     const int default_value)
{
  /* Set default value */
  *target = (int)default_value;

  add_generic_argument(parser, arg_name, ARG_Integer, (void *)target);
}

/** Register a boolean argument */
void
register_boolean_arg(arg_parser_t *parser, const char *arg_name, bool *target,
                     const bool default_value)
{
  /* Space to write reverse argument name, deallocation is done in
   * deallocate_arg_value */
  char *reverse_arg_name = (char*) malloc(strlen(arg_name) + 3);

  /* -no<arg> */
  strcpy(reverse_arg_name, "no");
  strcat(reverse_arg_name, arg_name);

  /* Set default value */
  *target = (bool)default_value;

  /* -<arg> */
  add_generic_argument(parser, arg_name, ARG_Boolean, (void *)target);
  /* -no<arg> */
  add_generic_argument(parser, reverse_arg_name, ARG_ReverseBoolean,
                       (void *)target);
}

/** Register a combines bool and string argument */
void
register_combined_bool_string_arg(arg_parser_t *parser, const char *arg_name,
                                  bool *bool_target, const char **string_target)
{
  /* Boolean target defaults to false (not set) */
  *bool_target = false;
  /* and string target defaults to NULL */
  *string_target = NULL;

  /* Store both pointers for argument processsing */
  bool_string_t *target = (bool_string_t*) malloc(sizeof(bool_string_t));
  target->bool_ptr = bool_target;
  target->string_ptr = string_target;

  add_generic_argument(parser, arg_name, ARG_CombinedBoolean, (void *)target);
}

/** Register "x" flag argument */
void
register_qflag_arg(arg_parser_t *parser, const char *arg_name, int *qflags,
                   const int qflags_size)
{
  /* Fill array with zeros */
  memset(qflags, 0, qflags_size * sizeof(int));

  add_generic_argument(parser, arg_name, ARG_QFlag, (void *)qflags);
}

/** Register "x" flag argument */
void
register_xflag_arg(arg_parser_t *parser, const char *arg_name, int *xflags)
{
  add_generic_argument(parser, arg_name, ARG_XFlag, (void *)xflags);
}

/** Register "y" flag argument */
void
register_yflag_arg(arg_parser_t *parser, const char *arg_name, int *xflags)
{
  add_generic_argument(parser, arg_name, ARG_YFlag, (void *)xflags);
}

/** Register verbosity argument */
void
register_inform_level_arg(arg_parser_t *parser, const char *arg_name,
                          inform_level_t *target,
                          const inform_level_t default_value)
{
  /* Set default value */
  *target = (inform_level_t)default_value;

  add_generic_argument(parser, arg_name, ARG_InformLevel, (void *)target);
}

/** Register "action list" argument */
void
register_action_map_arg(arg_parser_t *parser, const char *arg_name,
                        action_map_t *target, const action_map_t *input)
{
  action_map_bundle_t *value = (action_map_bundle_t*) malloc(
      sizeof(action_map_bundle_t));
  value->input = input;
  value->output = target;

  add_generic_argument(parser, arg_name, ARG_ActionMap, (void *)value);
}

/** Register input file name */
void
register_filename_arg(arg_parser_t *parser, const char **target)
{
  parser->input_file_name_ptr = target;
}

/** \brief Add a generic argument, specifying argument type
 *
 * \param arg_parser  Parser data structure
 * \param arg_name    Argument name (key to flags table)
 * \param value_type  Type of the value to store
 * \param value_ptr   Location to store value to
 */
static void
add_generic_argument(arg_parser_t *parser, const char *arg_name,
                     value_type_t value_type, void *value_ptr)
{
  /* Mapped value */
  value_t *value = NULL;
  /* Old (ignored) value */
  hash_data_t old_value = NULL;

  /* Check if this argument is already registered */
  if (hashmap_lookup(parser->flags, arg_name, &old_value)) {
    /* Compose & throw error */
    compose_and_throw("Argument already registered: ", arg_name);
  }

  /* Add value to flags hashmap */
  value = (value_t*) malloc(sizeof(value_t));
  value->type = value_type;
  value->location = value_ptr;
  hashmap_insert(parser->flags, arg_name, value);

  /* Put the same value into hashset tracking registered values (this time it is
   * the key) */
  if (value_type == ARG_ActionMap) {
    action_map_t *target = ((action_map_bundle_t *)value_ptr)->output;
    hashset_replace(parser->values, target);
  } else {
    hashset_replace(parser->values, value_ptr);
  }
}

/** Parse all arguments */
void
parse_arguments(const arg_parser_t *parser, int argc, char **argv)
{
  int argindex = 1;     /* Argument counter */
  char *next_string;    /* Next argument */
  int x_index, x_value; /* Index to xflags array and value to set there */

  /* Reset value hits */
  hashset_clear(parser->value_hits);

  /* Make sure we can set file name */
  if (!parser->input_file_name_ptr) {
    interr("Input file name is not registered", 0, ERR_Fatal);
  }

  /* First grab the source file name */
  if (*argv[1] != '-') {
    *parser->input_file_name_ptr = argv[1];
    argindex = 2;
  } else if (*argv[argc - 1] != '-') {
    *parser->input_file_name_ptr = argv[argc - 1];
    --argc;
  }

  /* Loop through provided arguments */
  while (argindex < argc) {
    value_t *value = NULL;
    char *arg = argv[argindex];

    /* All switches should start with a '-' */
    if (*arg != '-') {
      compose_and_throw("Unrecognized argument: ", arg);
    }
    ++arg; /* skip over '-' */

#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wcast-qual"
    /* All arguments need to be in the data structure */
    if (!hashmap_lookup(parser->flags, arg, (hash_data_t *)&value)) {
      if (parser->fail_on_unknown_args) {
        compose_and_throw("Unknown command line argument: ", arg);
      } else {
        /* Skip until next switch or end of argument list */
        while ((argindex < argc - 1) && (*argv[++argindex] != '-'))
          ;
        if (argindex == argc - 1)
          break;
        continue;
      }
    }
#pragma GCC diagnostic pop

    /* Parse argument type */
    switch (value->type) {
    case ARG_ActionMap: {
      const action_map_t *from = ((action_map_bundle_t *)value->location)->input;
      action_map_t *to = ((action_map_bundle_t *)value->location)->output;

      /* TODO parse lists of arguments */
      char *phase_string = next_value(argv, &argindex);
      char *dump_string = next_value(argv, &argindex);

      copy_action(from, dump_string, to, phase_string);
    } break;

    case ARG_Boolean:
      *((bool *)value->location) = true;
      break;

    case ARG_ReverseBoolean:
      *((bool *)value->location) = false;
      break;

    case ARG_String:
      next_string = next_value(argv, &argindex);
      if (!next_string)
        compose_and_throw("Missing value for -", arg);
      /* Change stored value to point to passed string */
      *((char **)value->location) = next_string;
      break;

    case ARG_StringList:
      /* Get argument value */
      next_string = next_value(argv, &argindex);
      if (!next_string)
        compose_and_throw("Missing value for -", arg);
      /* Set argument value */
      *((char **)value->location) = next_string;
      /* Point to the next value on the list */
      value->location = (void *)(((char **)value->location) + 1);
      /* Terminate the list */
      *((char **)value->location) = NULL;
      break;

    case ARG_InformLevel:
      next_string = next_value(argv, &argindex);
      if (!next_string)
        compose_and_throw("Missing value for -", arg);
      *((inform_level_t *)value->location) = strtoinform(next_string);
      break;

    case ARG_Integer:
      next_string = next_value(argv, &argindex);
      if (!next_string)
        compose_and_throw("Missing value for -", arg);
      *((int *)value->location) = (int)strtol(next_string, NULL, 10);
      break;

    case ARG_QFlag:
      next_string = next_value(argv, &argindex);

      if (!next_string)
        compose_and_throw("Missing value for -", arg);

      x_index = (int)strtol(next_string, NULL, 10);

      next_string = next_value(argv, &argindex);

      if (!next_string)
        compose_and_throw("Missing second value for -", arg);

      x_value = (int)strtol(next_string, NULL, 10);

      ((int *)value->location)[x_index] |= x_value;
      break;

    case ARG_XFlag:
      next_string = next_value(argv, &argindex);

      if (!next_string)
        compose_and_throw("Missing value for -", arg);
      x_index = (int)strtol(next_string, NULL, 10);

      next_string = next_value(argv, &argindex);

      if (next_string)
        x_value = (int)strtol(next_string, NULL, 0);
      else
        x_value = 1;

      set_xflag_value((int *)value->location, x_index, x_value);
      break;

    case ARG_YFlag:
      next_string = next_value(argv, &argindex);

      if (!next_string)
        compose_and_throw("Missing value for -", arg);
      x_index = (int)strtol(next_string, NULL, 10);

      next_string = next_value(argv, &argindex);

      if (next_string)
        x_value = (int)strtol(next_string, NULL, 0);
      else
        x_value = 1;

      unset_xflag_value((int *)value->location, x_index, x_value);
      break;

    case ARG_CombinedBoolean:
      *(((bool_string_t *)value->location)->bool_ptr) = true;
      next_string = next_value(argv, &argindex);
      if (next_string) {
        *(((bool_string_t *)value->location)->string_ptr) = next_string;
      }
      break;
    }

    /* Remember that the value as set */
    if (value->type == ARG_ActionMap) {
      action_map_t *target = ((action_map_bundle_t *)value->location)->output;
      hashset_replace(parser->value_hits, target);
    } else {
      hashset_replace(parser->value_hits, value->location);
    }

    ++argindex;
  }
}

/** \brief Compose and throw internal compiler error
 *
 * Produces internal compiler error with the message composed by concatenating
 * the two operands
 */
static void
compose_and_throw(const char *first_part, const char *second_part)
{
  char *msg = (char*) malloc(strlen(first_part) + strlen(second_part) + 1);
  strcpy(msg, first_part);
  strcat(msg, second_part);
  interr(msg, 0, ERR_Fatal);
}

/** \brief Point to next argument value
 *
 * If next argument in argument list does not start with a '-' return it and
 * advance argument counter, otherwise return NULL and keep counter the same
 *
 * \return          pointer to the next non-switch command line argument
 * \param argv      command line argument array
 * \param arg_index current index into the argument array
 */
static char *
next_value(char **argv, int *arg_index)
{
  if (*argv[*arg_index + 1] == '-')
    return NULL;

  *arg_index = *arg_index + 1;

  return argv[*arg_index];
}

/** \brief Convert a inform level string to corresponding constant */
static inform_level_t
strtoinform(const char *string)
{
  if (!strcmp("inform", string))
    return LV_Inform;
  if (!strcmp("severe", string))
    return LV_Severe;
  if (!strcmp("warn", string))
    return LV_Warn;
  if (!strcmp("fatal", string))
    return LV_Fatal;

  compose_and_throw("Unrecognized inform level: ", string);

  /* unreacheable */
  return LV_Inform;
}

/** Check if argument was found during parse */
bool
was_value_set(const arg_parser_t *parser, const void *location)
{
  /* Any value passed to this function need to be registered */
  if (!hashset_lookup(parser->values, location)) {
    compose_and_throw(__func__, ": value location not registered");
  }

  return (hashset_lookup(parser->value_hits, location));
}

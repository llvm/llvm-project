/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Debug action header file
 */

#ifndef DEBUG_ACTION_H
#define DEBUG_ACTION_H

/** \brief Forward-declared action map type
 */
typedef struct action_map_ action_map_t;

/** \brief Allocate and initialize action map data structure
 *
 * \param map Action map structure to initialize
 */
void create_action_map(action_map_t **map);

/** \brief Deallocate action map data structure
 */
void destroy_action_map(action_map_t **map);

/** \brief Add action to action map
 *
 * Multiple invocations of this function can add set multiple actions to the
 * same keyword as well as add the same action to multiple keyword.
 *
 * \param map     action map
 * \param keyword string to associate the action with
 * \param action  function pointer to invoke when argument is provided
 */
void add_action(action_map_t *map, const char *keyword, void (*action)(void));

/** \brief Copy an action from one map to another
 *
 * Take whatever is associated with a given keyword in one map and copy it to
 * another map.
 *
 * \param from         action map to copy from
 * \param keyword_from keyword in original action map
 * \param to           destination action map
 * \param keyword_to   new keyword for the destination
 */
void copy_action(const action_map_t *from, const char *keyword_from,
                 action_map_t *to, const char *keyword_to);

/** \brief Execute action(s) for a given keyword
 *
 * Order of execution is not guaranteed when multiple actions were added for
 * the same keyword. An action that was added multiple times executes multiple
 * times.
 *
 * \param map     action map
 * \param keyword the keyword to look up actions
 */
void execute_actions_for_keyword(action_map_t *map, const char *keyword);

#endif /* ndef DEBUG_ACTION_H */

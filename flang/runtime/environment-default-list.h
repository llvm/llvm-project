/*===-- runtime/environment-default-list.h --------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */

#ifndef FORTRAN_RUNTIME_ENVIRONMENT_DEFAULT_LIST_H_
#define FORTRAN_RUNTIME_ENVIRONMENT_DEFAULT_LIST_H_

/* Try to maintain C compatibility to make it easier to both define environment
 * defaults in non-Fortran main programs as well as pass through the environment
 * default list in C code.
 */

struct EnvironmentDefaultItem {
  const char *name;
  const char *value;
};

/* Default values for environment variables are packaged by lowering into an
 * instance of this struct to be read and set by the runtime.
 */
struct EnvironmentDefaultList {
  int numItems;
  const struct EnvironmentDefaultItem *item;
};

#endif /* FORTRAN_RUNTIME_ENVIRONMENT_DEFAULT_LIST_H_ */

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef RUNTIME_FLANG_GENLIST_H
#define RUNTIME_FLANG_GENLIST_H

/* generate list of cpu numbers */
struct cgrp *__fort_genlist(int nd,      /* number of dimensions */
                            int low,     /* lowest cpu number */
                            int cnts[],  /* counts per dimension */
                            int strs[]); /* strides per dimension */

#endif /* RUNTIME_FLANG_GENLIST_H */

/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef RUNTIME_FLANG_XFER_H
#define RUNTIME_FLANG_XFER_H

/* allocate channel structure */
struct chdr *__fort_allchn(struct chdr *cp, int dents, int sents, int cpus);

#endif /* RUNTIME_FLANG_XFER_H */

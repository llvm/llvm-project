//===-- Macros defined in stdio.h header file -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_STDIO_MACROS_H
#define LLVM_LIBC_MACROS_STDIO_MACROS_H

#include "../llvm-libc-types/FILE.h"

#ifdef __cplusplus
extern "C" FILE *stdin;
extern "C" FILE *stdout;
extern "C" FILE *stderr;
#else
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;
#endif

#ifndef stdin
#define stdin stdin
#endif

#ifndef stdout
#define stdout stdout
#endif

#ifndef stderr
#define stderr stderr
#endif

#ifndef EOF
#define EOF (-1)
#endif

#define BUFSIZ 1024

#define _IONBF 2
#define _IOLBF 1
#define _IOFBF 0

#ifndef SEEK_SET
#define SEEK_SET 0
#endif

#ifndef SEEK_CUR
#define SEEK_CUR 1
#endif

#ifndef SEEK_END
#define SEEK_END 2
#endif
/*
 * Derivation of L_tmpnam
 * ------------------------------------------------------------
 *
 * Generated pathnames have the form: /tmp/XXXXXXXXXXXXXX
 *   - "/tmp/" is a 5-byte prefix.
 *   - N random characters follow, drawn independently and uniformly from
 *     a 65-character alphabet (the POSIX portable filename character set)
 *   - 1 byte for the NULL terminator.
 * So: L_tmpnam = 5 + N + 1.
 *
 * Choosing N: we want the probability of two independently generated
 * suffixes colliding to stay below a target threshold P, even after up to
 * k calls to tmpnam() over the lifetime of a process.
 *
 * Let M = 65^N be the keyspace which is the total number of distinct
 * N-character suffixes that can be generated (NOT the number actually
 * generated; M is the size of the space they are drawn from).
 *
 * Among k calls, the number of distinct pairs of calls is:
 *     C(k, 2) = k(k-1)/2  ~=  k^2 / 2      (approximation valid for large k)
 *
 * Each individual pair collides (picks the identical suffix) with
 * probability 1/M, since each call draws independently and uniformly from
 * the M possible suffixes.
 *
 * Treating pairwise collisions as approximately independent low-probability
 * events, the probability that AT LEAST ONE collision occurs among all
 * pairs is approximately the sum over all pairs of the per-pair probability:
 *
 *     P  ~=  (k^2 / 2) * (1 / M)  =  k^2 / (2M)
 *
 * This is the standard birthday-bound approximation.
 *
 * Solving for the keyspace required to keep P under a chosen target, given
 * an assumed call-volume ceiling k:
 *
 *     M  >=  k^2 / (2P)
 *
 * Design inputs (stated, not borrowed):
 *     k = 10^6   (one million calls: a generous upper bound on how many
 *                 times a single long-running process could realistically
 *                 call tmpnam() in its lifetime)
 *     P = 10^-12 (one-in-a-trillion target collision probability)
 *
 * Required keyspace:
 *     M >= (10^6)^2 / (2 * 10^-12) = 5 x 10^23
 *
 * Solving 65^N >= 5x10^23 for N:
 *     N >= log_65(5x10^23) ~= 14 (round up)
 *
 * Verification with N = 14:
 *     M = 65^14 ~= 2.40 x 10^25
 *     Actual P at k = 10^6:  k^2 / (2M) ~= 2.08 x 10^-14
 *     (about 48x more conservative than the 10^-12 target -- the integer
 *      rounding of N gives us comfortable extra margin for free.)
 *
 * Therefore:
 *     N         = 14
 *     L_tmpnam  = 5 (prefix) + 14 (suffix) + 1 (NULL) = 20
 */
#ifndef L_tmpnam
#define L_tmpnam 20
#endif
/*
 * TMP_MAX:
 * ---------
 * TMP_MAX is a separate policy decision, that states the call-volume ceiling
 * for which we are willing to stand behind the P = 10^-12 collision-probability
 * guarantee derived above. Per POSIX, behavior beyond TMP_MAX calls in a single
 * process is implementation-defined; we simply decline to make any guarantee
 * past this point, even though the keyspace could technically support more.
 *
 *     TMP_MAX = 1,000,000
 *
 * This is chosen as a round, easily-reasoned-about figure equal to the k
 * used in the derivation above.
 *
 * Note on glibc's TMP_MAX = 238328: this value has no documented derivation.
 * A glibc/gnulib contributor publicly stated in 2001 that the figure's
 * origin is unknown and "as good as any other number larger than a couple
 * of thousand" (bug-textutils mailing list, Oct 26 2001:
 * https://lists.gnu.org/archive/html/bug-textutils/2001-10/msg00032.html).
 * We do not inherit this value; the derivation above is independent and
 * stated in full above.
 */
#ifndef TMP_MAX
#define TMP_MAX 1000000
#endif

#ifndef P_tmpdir
#define P_tmpdir "/tmp"
#endif

#endif // LLVM_LIBC_MACROS_STDIO_MACROS_H

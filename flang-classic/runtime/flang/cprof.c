/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \brief Profile initialization */
int
__fort_prof_init(void)
{
  return (0);
}

/** \brief Function entry  */
void
__fort_prof_function_entry(int line, int lines, int cline, char *func,
                           char *file, int funcl, int filel)
{
}

/** \brief Line entry  */
void
__fort_prof_line_entry(int line /* current line number */)
{
}

/** \brief Update start receive message stats
 * \param cpu: sending cpu
 * \param len: total length in bytes
 */
void
__fort_prof_recv(int cpu, long len)
{
}

/** \brief Update done receive message stats */
void
__fort_prof_recv_done(int cpu /* sending cpu */)
{
}

/** \brief Update start send message stats
 * \param cpu: receiving cpu
 * \param len: total length in bytes
 */
void
__fort_prof_send(int cpu, long len)
{
}

/** \brief Update done send message stats */
void
__fort_prof_send_done(int cpu /* receiving cpu */)
{
}

/** \brief Update start bcopy message stats
 * \param len: total length in bytes
 */
void
__fort_prof_copy(long len)
{
}

/** \brief Update done bcopy message stats */
void
__fort_prof_copy_done(void)
{
}

/** \brief Function exit  */
void
__fort_prof_function_exit(void)
{
}

/** \brief Profile termination */
void
__fort_prof_term(void)
{
}

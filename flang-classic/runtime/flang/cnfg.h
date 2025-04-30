/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 * \file
 * \brief cnfg.h -  declare external variables/arrays used by Fortran I/O
 */

/* declare structure which may alter the configuration */
typedef struct {
  const char *default_name; /* sprintf string for default file name */
  int true_mask;            /* 1 => odd is true, -1 => nonzero is true */
  int ftn_true;             /* -1 ==> VAX; 1 => unix */
} FIO_CNFG;

#if defined(_WIN64)

extern const char *__get_fio_cnfg_default_name(void);
extern int __get_fio_cnfg_true_mask(void);
extern int *__get_fio_cnfg_true_mask_addr(void);
extern int __get_fio_cnfg_ftn_true(void);
extern int *__get_fio_cnfg_ftn_true_addr(void);

#define GET_FIO_CNFG_DEFAULT_NAME __get_fio_cnfg_default_name()
#define GET_FIO_CNFG_TRUE_MASK __get_fio_cnfg_true_mask()
#define GET_FIO_CNFG_TRUE_MASK_ADDR __get_fio_cnfg_true_mask_addr()
#define GET_FIO_CNFG_FTN_TRUE __get_fio_cnfg_ftn_true()
#define GET_FIO_CNFG_FTN_TRUE_ADDR __get_fio_cnfg_ftn_true_addr()

#else

extern FIO_CNFG __fortio_cnfg_; /* ending '_' so it's accessible by the
                                * fortran programmer */

#define GET_FIO_CNFG_DEFAULT_NAME __fortio_cnfg_.default_name
#define GET_FIO_CNFG_TRUE_MASK __fortio_cnfg_.true_mask
#define GET_FIO_CNFG_TRUE_MASK_ADDR &__fortio_cnfg_.true_mask
#define GET_FIO_CNFG_FTN_TRUE __fortio_cnfg_.ftn_true
#define GET_FIO_CNFG_FTN_TRUE_ADDR &__fortio_cnfg_.ftn_true

#endif

extern void __fortio_scratch_name(char *, int);

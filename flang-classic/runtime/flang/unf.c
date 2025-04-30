/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Fortran unformatted I/O routines.
 */

#include <string.h>
#include "global.h"
#include "fioMacros.h"
#include "async.h"

static int __unf_init(bool, bool);
static int __unf_end(bool);
static int __usw_end(bool);
extern int __f90io_usw_read(int, long, int, char *, __CLEN_T);
extern int __f90io_usw_write(int, long, int, char *, __CLEN_T);
extern int __f90io_usw_end(void);
static int skip_to_nextrec(void);
static bool unf_fwrite(char *, size_t, size_t, FIO_FCB *);

/* define a few things for run-time tracing */
static int dbgflag;
#undef DBGBIT
#define DBGBIT(v) (LOCAL_DEBUG && (dbgflag & v))

#define IOBUFSIZE 4096

#define MAX_REC_SIZE (0x7fffffff - 8) /* -8 to allow for two length words */
#define CONT_FLAG 0x80000000          /* sign bit is continuation flag */
#define CONT_FLAG_SW 0x00000080       /* byte-swapped continuation flag */
#define TO_BE_CONTINUED TRUE

/* RCWSZ must match similar define in backspace.c */
#define RCWSZ sizeof(int)

#define UNF_ERR(e) return __fortio_error(e)

/* Globals to keep track of the current unformatted I/O operation.
 * These variables are initialized upon an __f90io_unf_init and used
 * and updated on __f90io_unf_writes and __f90io_unf_reads. All are
 * active till an __f90io_unf_end.  */

static FIO_FCB *Fcb;     /* pointer to the file control block */
static char *buf_ptr;    /* pointer to current location in buffer */
static size_t rw_size;   /* size of user-requested items (write only) */
static int rec_len;      /* record length */
static bool rec_in_buf;  /* true if variable len record in buffer; false
                            if access is direct. */
static bool read_flag;   /* true if a read, otherwise a write */
static bool io_transfer; /* indicates that init-end calls were made
                            with no intervening read or write calls */
static bool continued;   /* data requires multople records */
static bool async;       /* true if asynch i/o requested */
static bool actual_init;
static int has_same_fcb;

/*
 * define a structure which can be used to buffer a variable length
 * unformatted record (beginning record control word and data).
 * Needs to be defined in such a way that:
 * 1. count immediately precedes the data (buffer)
 * 2. the buffer is double word aligned (memset/memcpy performance)
 *
 * Notes:
 * 1. bcnt is the value of bytecnt at the time the record length word
 *    was written.  Used to catch the case of a writing a single item
 *    whose data does not fit in buf;  if this case occurs we do not
 *    need to seek to the beginning of the record to fill in the record
 *    length.
 * 2. During write operations, bytecnt (ByteCnt) is the length of the
 *    variable length record which is being written.  Rwsize is the number
 *    of bytes which have been buffered (in buf).
 * 3. During read operations, bytecnt is the the number of bytes which
 *    have been read.  Rwsize is not used.
 *
 * Optimizations notes:
 * 1. minimize system overhead (extraneous seeks)
 * 2. buffer items when writing.  For consecutive items, just a single memcpy
 *    is necessary; if buffer overflows, buffer is bypassed.
 *    For nonconsecutive items, buffer is always used to collect data.
 * 3. For read operations, data is always read into user's variables.  If
 *    items are consecutive, a single read is performed.
 *
 */

typedef struct {
  union {
    double dalign;
    struct {
      int bcnt;    /* pad and value of bytecnt when written */
      int bytecnt; /* # of bytes read/written */
    } s;
  } u;
  char buf[IOBUFSIZE]; /* buffer */
  int pad;             /* just in case we need trailing count */
} unf_rec_struct;

static unf_rec_struct unf_rec;

typedef struct {
  FIO_FCB *Fcb;
  char *buf_ptr;
  int rw_size;
  int rec_len;
  bool rec_in_buf;
  bool read_flag;
  bool io_transfer;
  bool continued;
  bool async;
  int has_same_fcb;
  unf_rec_struct unf_rec;
} G;

#define GBL_SIZE 5

static G static_gbl[GBL_SIZE];
static G *gbl = &static_gbl[0];
static G *gbl_head = &static_gbl[0];
static int gbl_avl = 0;
static int gbl_size = GBL_SIZE;

#define WRITE_UNF_LEN                                                          \
  (unf_fwrite((char *)&unf_rec.u.s.bytecnt, RCWSZ, 1, Fcb) != TRUE)
#define WRITE_UNF_REC                                                          \
  (unf_fwrite((char *)&unf_rec.u.s.bytecnt, rw_size + RCWSZ, 1, Fcb) != TRUE)
#define WRITE_UNF_BUF write_unf_buf()

static int
adjust_fpos(FIO_FCB *cur_file, long offset, int whence)
{
  int retval = 0;

  if (cur_file->asy_rw) {
    Fio_asy_fseek(cur_file->asyptr, offset, whence);
  } else {
    retval = __io_fseek(cur_file->fp, offset, whence);
  }
  return retval;
}

static void
save_gbl()
{
  int buffOffset;
  if (gbl_avl) {
    gbl->Fcb = Fcb;
    gbl->rw_size = rw_size;
    gbl->rec_len = rec_len;
    gbl->rec_in_buf = rec_in_buf;
    gbl->read_flag = read_flag;
    gbl->io_transfer = io_transfer;
    gbl->continued = continued;
    gbl->async = async;
    memcpy(&(gbl->unf_rec), &unf_rec, sizeof(unf_rec_struct));
    buffOffset = buf_ptr - unf_rec.buf;
    gbl->buf_ptr = gbl->unf_rec.buf + buffOffset;
    gbl->has_same_fcb = has_same_fcb;
  }
}

static void
restore_gbl()
{
  int buffOffset;
  if (gbl_avl) {
    Fcb = gbl->Fcb;
    rw_size = gbl->rw_size;
    rec_len = gbl->rec_len;
    rec_in_buf = gbl->rec_in_buf;
    read_flag = gbl->read_flag;
    io_transfer = gbl->io_transfer;
    continued = gbl->continued;
    async = gbl->async;
    memcpy(&unf_rec, &(gbl->unf_rec), sizeof(unf_rec_struct));
    buffOffset = gbl->buf_ptr - gbl->unf_rec.buf;
    buf_ptr = unf_rec.buf + buffOffset;
    has_same_fcb = gbl->has_same_fcb;
  }
}

static void
allocate_new_gbl()
{
  G *tmp_gbl;
  if (gbl_avl >= gbl_size) {
    if (gbl_size == GBL_SIZE) {
      gbl_size = gbl_size + 15;
      tmp_gbl = (G *)malloc(sizeof(G) * gbl_size);
      memcpy(tmp_gbl, gbl_head, sizeof(G) * gbl_avl);
      gbl_head = tmp_gbl;
    } else {
      gbl_size = gbl_size + 15;
      gbl_head = (G *)realloc(gbl_head, sizeof(G) * gbl_size);
    }
  }
  gbl = &gbl_head[gbl_avl];
  memset(gbl, 0, sizeof(G));
  ++gbl_avl;
}

static void
free_gbl()
{
  --gbl_avl;
  if (gbl_avl <= 0)
    gbl_avl = 0;
  if (gbl_avl == 0)
    gbl = &gbl_head[0];
  else
    gbl = &gbl_head[gbl_avl - 1];
}

/** \brief
 * Write the contents of unf_rec.buf; return TRUE if an
 * error occurred.
 */
static bool
write_unf_buf()
{
  if (rw_size && (unf_fwrite(unf_rec.buf, rw_size, 1, Fcb) != TRUE))
    return TRUE;
  return FALSE;
}

static bool
unf_fwrite(char *buf, size_t size, size_t num, FIO_FCB *fcb)
{
  if (fcb->asy_rw) {
    /* Do this write asynchronously. */
    return (Fio_asy_write(fcb->asyptr, buf, size * num) == 0);
  } else {
    /* Do this write "normally." */
    return (FWRITE(buf, size, num, fcb->fp) == num);
  }
  return FALSE;
}

/* initialize asynch i/o, called before Fio_unf_init */

int
ENTF90IO(UNF_ASYNCA, unf_asynca)(DCHAR(asy), __INT_T *id DCLEN64(asy))
{
  async = 0;
  if (!ISPRESENTC(asy))
    return 0;
  if (__fortio_eq_str(CADR(asy), CLEN(asy), "YES")) {
    if (id != NULL)
      *id = 0;
    async = 1;
    return 0;
  }
  if (__fortio_eq_str(CADR(asy), CLEN(asy), "NO")) {
    return (0);
  }
  UNF_ERR(FIO_ESPEC);
}
/* 32 bit CLEN version */
int
ENTF90IO(UNF_ASYNC, unf_async)(DCHAR(asy), __INT_T *id DCLEN(asy))
{
  return ENTF90IO(UNF_ASYNCA, unf_asynca)(CADR(asy), id, (__CLEN_T)CLEN(asy));
}

/* --------------------------------------------------------------------- */

/** \brief Initialize global flags to prepare for unformatted I/O, and if the
 * file isn't opened, open it (if possible).  */
int
__f90io_unf_init(__INT_T *read,   /* TRUE indicates READ statement. */
                 __INT_T *unit,   /* unit number. */
                 __INT_T *rec,    /* record number for direct access */
                 __INT_T *bitv,   /* same as for ENTF90IO(_open_). */
                 __INT_T *iostat) /* same as for ENTF90IO(open_). */
{
  int s = 0;

  save_gbl();

  if (*read)
    __fortio_errinit03(*unit, *bitv, iostat, "unformatted read");
  else
    __fortio_errinit03(*unit, *bitv, iostat, "unformatted write");

  allocate_new_gbl();
  Fcb = __fortio_rwinit(*unit, FIO_UNFORMATTED, rec, 1 - *read);

  if (Fcb == NULL) {
    if (fioFcbTbls.eof)
      return EOF_FLAG;
    /* TBD - does there need to be fioFcbTbls.eor */
    return ERR_FLAG;
  }
  gbl->Fcb = Fcb;
  continued = FALSE;

  actual_init = TRUE;
  s = __unf_init(*read, Fcb->byte_swap);
  actual_init = FALSE;
  return s;
}

/** \brief  Initialize global flags to prepare for unformatted I/O;
 *  \param read      - TRUE indicates READ statement.
 *  \param byte_swap - TRUE if byteswap I/O
 *
 *  shared by __f90io_unf_init() & __f90io_usw_init().
 */
static int
__unf_init(bool read, bool byte_swap)
{
  int a, i; /* async flag saved here */
  int buffOffset;
  G *tmp_gbl;

  a = async;
  async = 0;

  read_flag = read;

  /* recursive i/o and check all recursive fcb, starting from latest recursive
   */
  tmp_gbl = NULL;
  if (actual_init && gbl_avl > 1) {
    for (i = gbl_avl - 2; i >= 0; --i) {
      if (gbl_head[i].Fcb == Fcb) {
        tmp_gbl = &gbl_head[i];
        break;
      }
    }
  }
  if (tmp_gbl) {

    /* copy all tmp_gbl to global static variables */
    memcpy(&unf_rec, &tmp_gbl->unf_rec, sizeof(unf_rec_struct));
    buffOffset = (tmp_gbl->buf_ptr) - (tmp_gbl->unf_rec.buf);
    buf_ptr = unf_rec.buf + buffOffset;
    rec_len = tmp_gbl->rec_len;
    io_transfer = tmp_gbl->io_transfer;
    rec_in_buf = tmp_gbl->rec_in_buf;
    rw_size = tmp_gbl->rw_size;
    continued = tmp_gbl->continued;
    has_same_fcb = 1;
    return 0;

  } else {
    io_transfer = FALSE;
    rec_in_buf = FALSE;
    buf_ptr = unf_rec.buf;
    unf_rec.u.s.bytecnt = 0;
    if (actual_init)
      has_same_fcb = 0;
  }

  if (Fcb->acc == FIO_DIRECT)
    rec_len = Fcb->reclen;
  else if (!Fcb->binary && read) {
    /* sequential access - read reclen word */
    if (!continued)
      Fcb->nextrec++;
    if (__io_fread(&rec_len, RCWSZ, 1, Fcb->fp) != 1) {
      if (__io_feof(Fcb->fp))
        UNF_ERR(FIO_EEOF);
      UNF_ERR(__io_errno());
    }
    if (byte_swap)
      __fortio_swap_bytes((char *)&rec_len, __INT, 1);
    if (!f90_old_huge_rec_fmt()) {
      continued = (rec_len < 0);
      if (continued)
        rec_len = -rec_len;
    } else {
      continued = ((rec_len & CONT_FLAG) != 0);
      rec_len &= ~CONT_FLAG;
    }
  }

  if (a) { /* starting async i/o? */
    if (Fcb->asyptr == (void *)0) {
      UNF_ERR(FIO_EASYNC);
    }
    if (Fio_asy_enable(Fcb->asyptr) == -1) {
      Fcb->asy_rw = 0;
      UNF_ERR(__io_errno());
    }
    Fcb->asy_rw = 1;
  } else if (Fcb->asy_rw) { /* no, already async? */
    Fcb->asy_rw = 0;
    if (Fio_asy_disable(Fcb->asyptr) == -1) {
      UNF_ERR(__io_errno());
    }
  }

  if (!read) {
    if (Fcb->acc != FIO_DIRECT && !tmp_gbl)
      rec_in_buf = TRUE;
    if (!tmp_gbl)
      rw_size = 0;
  }

  return 0;
}

__INT_T
ENTF90IO(UNF_INIT, unf_init)
(__INT_T *read,   /* TRUE indicates READ statement. */
 __INT_T *unit,   /* unit number. */
 __INT_T *rec,    /* record number for direct access */
 __INT_T *bitv,   /* same as for ENTF90IO(open). */
 __INT_T *iostat) /* same as for ENTF90IO(open). */
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_unf_init(read, unit, rec, bitv, iostat);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

/* ----------------------------------------------------------------------- */

/** \brief  Read/copy data from an unformatted record file. */
int
__f90io_unf_read(int type,    /* Type of data */
                 long length, /* number of items of specified type
                               * to read.  May be <= 0 */
                 int stride,  /* distance in bytes between items */
                 char *item,  /* where to xfer data */
                 __CLEN_T item_length)
{
  size_t nbytes; /* number of bytes to read from current record */
  size_t resid;  /* number of bytes to read from following records */
  int offset;    /* offset into item */
  int ret_val = 0;

  /* first check for errors: */
  if (fioFcbTbls.eof) {
    ret_val = EOF_FLAG;
    goto unfr_err;
  }
  if (fioFcbTbls.error || !Fcb) {
    ret_val = ERR_FLAG;
    goto unfr_err;
  }

  if (Fcb->byte_swap) {
    return __f90io_usw_read(type, length, stride, item, item_length);
  }

  assert(item != NULL);

  if (length <= 0) /* treat this call as no-op */
    return 0;
  io_transfer = TRUE;

  nbytes = (size_t)length * item_length;
  offset = 0;

unf_read_do_resid:
  resid = 0;
  if (!Fcb->binary) {
    if (unf_rec.u.s.bytecnt + nbytes > (size_t)rec_len) {
      /* Not enough data in the current record to satisfy the request.
         If data continues in the next record, get the residual there. */
      if (!continued) {
        (void)skip_to_nextrec();
        ret_val = __fortio_error(FIO_ETOOBIG);
        goto unfr_err;
      }
      resid = unf_rec.u.s.bytecnt + nbytes - rec_len;
      nbytes -= resid;
    }

    if (rec_len == 0)
      return 0;
  }

  /* read directly into item if possible  (consecutive items) */

  if ((stride == 0) || ((__CLEN_T)stride == item_length)) {
    unf_rec.u.s.bytecnt += nbytes;
    if (Fcb->asy_rw) { /* XXXXXX XX */
      if (Fio_asy_read(Fcb->asyptr, item, nbytes) == -1) {
        ret_val = __fortio_error(__io_errno());
        goto unfr_err;
      }
      return (0);
    }
    if (__io_fread(item, nbytes, 1, Fcb->fp) != 1) {
      if (__io_feof(Fcb->fp)) {
        ret_val = __fortio_error(FIO_EEOF);
        if (Fcb->partial) {
          Fcb->partial = 0;
          ret_val = __fortio_error(FIO_EDREAD);
        } else {
          ret_val = __fortio_error(FIO_EEOF);
        }
      } else
        ret_val = __fortio_error(__io_errno());
      goto unfr_err;
    }

    if (resid > 0) {
      /* There is more data in the next record. */
      if ((ret_val = __unf_end(TO_BE_CONTINUED)) != 0) {
        goto unfr_err;
      }
      if ((ret_val = __unf_init(read_flag, Fcb->byte_swap)) != 0) {
        goto unfr_err;
      }
      item += nbytes;
      nbytes = resid;
      io_transfer = TRUE;
      goto unf_read_do_resid;
    }
    return 0;
  }

  /* copy 'length' items from stream into 'item', skipping by 'stride' */

  while (nbytes > 0) {
    int read_length;

    /* Read the lesser of
            bytes remaining in record (nbytes), or
            bytes needed to fill the item (item_length - offset) */
    read_length =
        (nbytes < item_length - offset ? nbytes : item_length - offset);
    if (__io_fread(item + offset, read_length, 1, Fcb->fp) != 1) {
      if (__io_feof(Fcb->fp))
        ret_val = __fortio_error(FIO_EEOF);
      else
        ret_val = __fortio_error(__io_errno());
      goto unfr_err;
    }
    unf_rec.u.s.bytecnt += read_length;
    nbytes -= read_length;
    offset += read_length;
    if ((__CLEN_T)offset == item_length) {
      item += stride;
      offset = 0;
    }
  }
  if (resid > 0) {
    /* There is more data in the next record. */
    if ((ret_val = __unf_end(TO_BE_CONTINUED)) != 0) {
      goto unfr_err;
    }
    if ((ret_val = __unf_init(read_flag, Fcb->byte_swap)) != 0) {
      goto unfr_err;
    }
    nbytes = resid;
    io_transfer = TRUE;
    goto unf_read_do_resid;
  }

  return 0;
unfr_err:
  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return ret_val;
}

__INT_T
ENTF90IO(UNF_READA, unf_reada)
(__INT_T *type,   /* Type of data */
 __INT_T *count,  /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN64(item))
{
  int s = 0;
  int ioproc;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_unf_read(*type, *count, *stride, CADR(item), len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, CADR(item), *count, *stride, *type, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(UNF_READ, unf_read)
(__INT_T *type,   /* Type of data */
 __INT_T *count,  /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN(item))
{
  return ENTF90IO(UNF_READA, unf_reada)(type, count, stride, CADR(item),
                                        (__CLEN_T)CLEN(item));
}

/** \brief same as unf_read, but item may be array - for unf_read, the compiler
 * scalarizes.
 */
__INT_T
ENTF90IO(UNF_READ_AA, unf_read_aa)
(__INT_T *type,   /* Type of data */
 __INT_T *count,  /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN64(item))
{
  int s = 0;
  int ioproc;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_unf_read(*type, *count, *stride, CADR(item), len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, CADR(item), *count, *stride, *type, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(UNF_READ_A, unf_read_a)
(__INT_T *type,   /* Type of data */
 __INT_T *count,  /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN(item))
{
  return ENTF90IO(UNF_READ_AA, unf_read_aa)(type, count, stride, CADR(item),
                                            (__CLEN_T)CLEN(item));
}

__INT_T
ENTF90IO(UNF_READ64_AA, unf_read64_aa)
(__INT_T *type,   /* Type of data */
 __INT8_T *count, /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN64(item))
{
  int s = 0;
  int ioproc;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_unf_read(*type, *count, *stride, CADR(item), len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, CADR(item), *count, *stride, *type, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(UNF_READ64_A, unf_read64_a)
(__INT_T *type,   /* Type of data */
 __INT8_T *count, /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN(item))
{
  return ENTF90IO(UNF_READ64_AA, unf_read64_aa)(type, count, stride, CADR(item),
                                                (__CLEN_T)CLEN(item));
}

/* Read/copy bytes from an unformatted record file; used when the item
 * is an aggregate. */

__INT_T
ENTF90IO(BYTE_READA, byte_reada)
(__INT_T *count,        /* number of items of specified type
                         * to read.  May be <= 0 */
 __INT_T *stride,       /* distance in bytes between items */
 char *item,            /* where to xfer data */
 __CLEN_T *item_length) /* number of bytes */
{
  int ioproc;
  int s = 0;

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_unf_read(__STR, *count, *stride, item, *item_length);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, item, *count, *stride, __STR, *item_length);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(BYTE_READ, byte_read)
(__INT_T *count,       /* number of items of specified type
                        * to read.  May be <= 0 */
 __INT_T *stride,      /* distance in bytes between items */
 char *item,           /* where to xfer data */
 __INT_T *item_length) /* number of bytes */
{
  return ENTF90IO(BYTE_READA, byte_reada)(count, stride, item,
                                          (__CLEN_T *)item_length);
}

__INT_T
ENTF90IO(BYTE_READ64A, byte_read64a)
(__INT8_T *count,       /* number of items of specified type
                         * to read.  May be <= 0 */
 __INT_T *stride,       /* distance in bytes between items */
 char *item,            /* where to xfer data */
 __CLEN_T *item_length) /* number of bytes */
{
  int ioproc;
  int s = 0;
  /*
   * NOTE: At this time, BYTE_READ64 is just a a byte stream read, i.e.,
   * count is the total number of bytes, stride should be zero, and
   * item_length is 1.
   */
  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_unf_read(__STR, *count, *stride, item, *item_length);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, item, *count, *stride, __STR, *item_length);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(BYTE_READ64, byte_read64)
(__INT8_T *count,      /* number of items of specified type
                        * to read.  May be <= 0 */
 __INT_T *stride,      /* distance in bytes between items */
 char *item,           /* where to xfer data */
 __INT_T *item_length) /* number of bytes */
{
  return ENTF90IO(BYTE_READ64A, byte_read64a)(count, stride, item,
                                              (__CLEN_T *)item_length);
}

/* ----------------------------------------------------------------- */

int
__f90io_unf_write(int type,   /* data type of data (see above). */
                  long count, /* number of items of specified type
                               * to write.  May be <= 0 */
                  int stride, /* distance in bytes between items. */
                  char *item, /* where to get data from */
                  __CLEN_T item_length)
{
  long i;        /* loop index */
  __CLEN_T j;    /* loop index */
  size_t nbytes; /* # of bytes to write for this call */
  int ret_val;

  /* first check for errors: */
  assert(fioFcbTbls.eof == 0);
  if (fioFcbTbls.error || !Fcb) {
    ret_val = ERR_FLAG;
    goto unf_write_err;
  }

  if (Fcb->byte_swap) {
    return __f90io_usw_write(type, count, stride, item, item_length);
  }

  assert(item != NULL);

  if (count <= 0)
    return 0;
  io_transfer = TRUE;

  nbytes = (size_t)count * item_length;
  if (Fcb->acc == FIO_DIRECT &&
      unf_rec.u.s.bytecnt + nbytes > (size_t)rec_len) {
    ret_val = __fortio_error(FIO_ETOOBIG);
    goto unf_write_err;
  }

  if (item_length == (size_t)stride || count == 1) {
    /*  optimize if we have consecutive items  */
    size_t resid; /* # of bytes spilled to next record */

  unf_write_do_resid:
    if (unf_rec.u.s.bytecnt + nbytes > MAX_REC_SIZE) {
      resid = unf_rec.u.s.bytecnt + nbytes - MAX_REC_SIZE;
      nbytes -= resid;
    } else {
      resid = 0;
    }
    unf_rec.u.s.bytecnt += nbytes;
    if (rw_size + nbytes > IOBUFSIZE || resid > 0) {
      if (DBGBIT(0x4))
        __io_printf(
            ("unit stride flush, rw_size=%" GBL_SIZE_T_FORMAT ", in_buf:%d\n"),
            rw_size, rec_in_buf);
      if (rec_in_buf) {
        if (!Fcb->binary) {
          if (WRITE_UNF_REC) {
            ret_val = __fortio_error(__io_errno());
            goto unf_write_err;
          }
        } else {
          if (WRITE_UNF_BUF) {
            ret_val = __fortio_error(__io_errno());
            goto unf_write_err;
          }
        }
        rec_in_buf = FALSE;
        unf_rec.u.s.bcnt = unf_rec.u.s.bytecnt;
      } else {
        if (WRITE_UNF_BUF) {
          ret_val = __fortio_error(__io_errno());
          goto unf_write_err;
        }
      }
      if (DBGBIT(0x4))
        __io_printf("unit stride write, nbytes=%" GBL_SIZE_T_FORMAT "\n",
                    nbytes);
      if (unf_fwrite(item, nbytes, 1, Fcb) != TRUE) {
        ret_val = __fortio_error(__io_errno());
        goto unf_write_err;
      }
      rw_size = 0;
      buf_ptr = unf_rec.buf;
      if (resid > 0) {
        if ((ret_val = __unf_end(TO_BE_CONTINUED)) != 0) {
          goto unf_write_err;
        }
        if ((ret_val = __unf_init(read_flag, Fcb->byte_swap)) != 0) {
          goto unf_write_err;
        }
        item += nbytes;
        nbytes = resid;
        io_transfer = TRUE;
        goto unf_write_do_resid;
      }
      return 0;
    }
    if (DBGBIT(0x4))
      __io_printf("unit stride copy, nbytes=%" GBL_SIZE_T_FORMAT
                  ", rw_size=%" GBL_SIZE_T_FORMAT ", in_buf:%d\n",
                  nbytes, rw_size, rec_in_buf);
    (void)memcpy(buf_ptr, item, nbytes);
    buf_ptr += nbytes;
    rw_size += nbytes;
    return 0;
  }

  if (Fcb->asy_rw) { /* stop any async i/o */
    Fcb->asy_rw = 0;
    if (Fio_asy_disable(Fcb->asyptr) == -1) {
      ret_val = __fortio_error(__io_errno());
      goto unf_write_err;
    }
  }

  /* copy 'count' items from 'item' into buffer, skipping by stride  */

  for (i = 0; i < count; i++, item += (stride - item_length)) {
    int rec_full;

    if ((unf_rec.u.s.bytecnt + item_length) > MAX_REC_SIZE)
      rec_full = TRUE;
    else {
      rec_full = FALSE;
      unf_rec.u.s.bytecnt += item_length;
    }
    if ((rw_size + item_length) >= IOBUFSIZE || rec_full) {
      if (DBGBIT(0x4))
        __io_printf("non-unit stride flush, nbytes=%" GBL_SIZE_T_FORMAT
                    ", in_buf:%d\n",
                    rw_size, rec_in_buf);
      if (rec_in_buf) {
        if (!Fcb->binary) {
          if (WRITE_UNF_REC) {
            ret_val = __fortio_error(__io_errno());
            goto unf_write_err;
          }
        } else {
          if (WRITE_UNF_BUF) {
            ret_val = __fortio_error(__io_errno());
            goto unf_write_err;
          }
        }
        rec_in_buf = FALSE;
        unf_rec.u.s.bcnt = unf_rec.u.s.bytecnt;
      } else {
        if (WRITE_UNF_BUF) {
          ret_val = __fortio_error(__io_errno());
          goto unf_write_err;
        }
      }
      rw_size = 0;
      buf_ptr = unf_rec.buf;
      if (rec_full) {
        /* Start a new record. */
        if ((ret_val = __unf_end(TO_BE_CONTINUED)) != 0) {
          goto unf_write_err;
        }
        if ((ret_val = __unf_init(read_flag, Fcb->byte_swap)) != 0) {
          goto unf_write_err;
        }
        /* Current item was not written; try again. */
        i--;
        item -= (stride - item_length);
        io_transfer = TRUE;
        continue;
      }
      if (item_length > IOBUFSIZE) {
        if (unf_fwrite(item, item_length, 1, Fcb) != TRUE) {
          ret_val = __fortio_error(__io_errno());
          goto unf_write_err;
        }
        item += item_length;
        continue;
      }
    }
    /* ??? one copy per byte.  Maybe optimize later */
    for (j = 0; j < item_length; j++)
      *buf_ptr++ = *item++;

    rw_size += item_length;
  }
  return 0;

unf_write_err:
  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return ret_val;
}

__INT_T
ENTF90IO(UNF_WRITEA, unf_writea)
(__INT_T *type,   /* data type of data (see above). */
 __INT_T *count,  /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN64(item))
{
  int s = 0;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_unf_write(*type, *count, *stride, CADR(item), len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(UNF_WRITE, unf_write)
(__INT_T *type,   /* data type of data (see above). */
 __INT_T *count,  /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN(item))
{
  return ENTF90IO(UNF_WRITEA, unf_writea)(type, count, stride, CADR(item),
                                          (__CLEN_T)CLEN(item));
}

/** \brief same as unf_write, but item may be array - for unf_write, the
 * compiler scalarizes.
 */
__INT_T
ENTF90IO(UNF_WRITE_AA, unf_write_aa)
(__INT_T *type,   /* data type of data (see above). */
 __INT_T *count,  /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN64(item))
{
  int s = 0;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_unf_write(*type, *count, *stride, CADR(item), len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(UNF_WRITE_A, unf_write_a)
(__INT_T *type,   /* data type of data (see above). */
 __INT_T *count,  /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN(item))
{
  return ENTF90IO(UNF_WRITE_AA, unf_write_aa)(type, count, stride, CADR(item),
                                              (__CLEN_T)CLEN(item));
}

__INT_T
ENTF90IO(UNF_WRITE64_AA, unf_write64_aa)
(__INT_T *type,   /* data type of data (see above). */
 __INT8_T *count, /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN64(item))
{
  int s = 0;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_unf_write(*type, *count, *stride, CADR(item), len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(UNF_WRITE64_A, unf_write64_a)
(__INT_T *type,   /* data type of data (see above). */
 __INT8_T *count, /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN(item))
{
  return ENTF90IO(UNF_WRITE64_AA, unf_write64_aa)(
      type, count, stride, CADR(item), (__CLEN_T)CLEN(item));
}

/** \brief Write bytes to an unformatted record file; used when the item is an
 * aggregate. */

__INT_T
ENTF90IO(BYTE_WRITEA, byte_writea)
(__INT_T *count,        /* number of items of specified type
                         * to write.  May be <= 0 */
 __INT_T *stride,       /* distance in bytes between items */
 char *item,            /* where to get data from */
 __CLEN_T *item_length) /* number of bytes */
{
  int s = 0;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_unf_write(__STR, *count, *stride, item, *item_length);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(BYTE_WRITE, byte_write)
(__INT_T *count,       /* number of items of specified type
                        * to write.  May be <= 0 */
 __INT_T *stride,      /* distance in bytes between items */
 char *item,           /* where to get data from */
 __INT_T *item_length) /* number of bytes */
{
  return ENTF90IO(BYTE_WRITEA, byte_writea)(count, stride, item,
                                            (__CLEN_T *)item_length);
}

__INT_T
ENTF90IO(BYTE_WRITE64A, byte_write64a)
(__INT8_T *count,       /* number of items of specified type
                         * to write.  May be <= 0 */
 __INT_T *stride,       /* distance in bytes between items */
 char *item,            /* where to get data from */
 __CLEN_T *item_length) /* number of bytes */
{
  int s = 0;
  /*
   * NOTE: At this time, BYTE_WRITE64 is just a a byte stream write, i.e.,
   * count is the total number of bytes, stride should be zero, and
   * item_length is 1.
   *
   * With this in mind, simply passing a stride of 0 will have the effect
   * of __f90io_unf_write() looping count times on a write of 1 byte
   * WITHOUT incrementing the item pointer -- trick __f90io_unf_write into
   * a single byte stream write by making stride the item_length.
   */
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_unf_write(__STR, *count, *item_length, item, *item_length);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(BYTE_WRITE64, byte_write64)
(__INT8_T *count,      /* number of items of specified type
                        * to write.  May be <= 0 */
 __INT_T *stride,      /* distance in bytes between items */
 char *item,           /* where to get data from */
 __INT_T *item_length) /* number of bytes */
{
  return ENTF90IO(BYTE_WRITE64A, byte_write64a)(count, stride, item,
                                                (__CLEN_T *)item_length);
}

/* -------------------------------------------------------------------- */

/** \brief Finish up unformatted read or write.
 *
 * If current I/O is a read, * write the current buffer to the file.  Whether
 * a read or a write, free the buffer. */
int
__f90io_unf_end(void)
{
  if (fioFcbTbls.error)
    return ERR_FLAG;
  if (fioFcbTbls.eof || !Fcb)
    return EOF_FLAG;
  if (has_same_fcb)
    return 0;

  if (Fcb->byte_swap)
    return __f90io_usw_end();

  return __unf_end(!TO_BE_CONTINUED);
}

static int
__unf_end(bool to_be_continued)
{
  int ret_err;

  /* if read and direct (variable length) then seek past the trailing
   * 4-byte integer that indicates the record's size so the next read/write
   * will be at the next record: */

  /* From this point on, async i/o has been disabled, so use "normal" fseek
   * calls.
   */
  if (read_flag) {
    if (Fcb->binary) {
      Fcb->coherent = 0;
      return 0;
    }
    if (!io_transfer) {
      /*
       * read with no input list-- seek past current record.
       * this only happens if we have a READ statement without any
       * items.
       */
      if (Fcb->acc != FIO_DIRECT)
        ret_err = __io_fseek(Fcb->fp, (seekoffx_t)rec_len + RCWSZ, SEEK_CUR);
      else
        ret_err = __io_fseek(Fcb->fp, (seekoffx_t)rec_len, SEEK_CUR);
      if (ret_err)
        UNF_ERR(__io_errno());
      Fcb->coherent = 0;
      return 0;
    }

    ret_err = skip_to_nextrec();
    if (ret_err)
      UNF_ERR(ret_err);

    /* If we want to use data continued into the next record, we're done.
       Else we need to seek to the actual end of data by watching the
       continue flags. */
    if (to_be_continued)
      return 0;
    while (continued) {
      if (__io_fread(&rec_len, 4, 1, Fcb->fp) != 1)
        UNF_ERR(__io_errno());
      if (!f90_old_huge_rec_fmt()) {
        if (__io_fseek(Fcb->fp, -rec_len + 4, SEEK_CUR))
          continued = (rec_len < 0);
      } else {
        if (__io_fseek(Fcb->fp, (rec_len &= ~CONT_FLAG) + 4, SEEK_CUR))
          continued = (rec_len & CONT_FLAG);
      }
      UNF_ERR(__io_errno());
    }
    return 0;
  }

  /*
   * Case 2:  Write statment.  write the buffer to the file.  check if direct
   * (variable length), and if so poke in the reclen to the first and
   * last 4 bytes of the buffer else pad the rest of the buffer with
   * zeroes
   */
  if (!io_transfer) {
    if (Fcb->acc != FIO_DIRECT) { /* write 0 length record */
      if (Fcb->binary)
        return 0;
      ret_err = __fortio_zeropad(Fcb->fp, RCWSZ << 1);
      if (ret_err != 0)
        UNF_ERR(ret_err);
      return 0;
    }
    if (!has_same_fcb)
      rw_size = 0;
  }

  if (rec_in_buf) {
    if (has_same_fcb)
      return 0;
    if (Fcb->binary) {
      if (WRITE_UNF_BUF)
        UNF_ERR(__io_errno());
      return 0;
    }
    if (WRITE_UNF_REC)
      UNF_ERR(__io_errno());
    if (WRITE_UNF_LEN)
      UNF_ERR(__io_errno());
    return 0;
  }

  if (!has_same_fcb)
    if (WRITE_UNF_BUF)
      UNF_ERR(__io_errno());
  if (!has_same_fcb) {
    rw_size = 0;
    buf_ptr = unf_rec.buf;
  }

  if (Fcb->acc != FIO_DIRECT) {
    if (Fcb->binary)
      return 0;
    if (unf_rec.u.s.bcnt != unf_rec.u.s.bytecnt || to_be_continued) {
      int bytecnt = unf_rec.u.s.bytecnt;
      /* If this record's data is to be continued in the next record,
         set the continuation bit in this record's leading length word. */
      if (to_be_continued) {
        if (!f90_old_huge_rec_fmt()) {
          unf_rec.u.s.bytecnt = -unf_rec.u.s.bytecnt;
        } else {
          unf_rec.u.s.bytecnt |= CONT_FLAG;
        }
      }
      /* seek to record's beginning length word */
      if (adjust_fpos(Fcb, (seekoffx_t)(-bytecnt) - (seekoffx_t)(RCWSZ),
                      SEEK_CUR) != 0)
        UNF_ERR(__io_errno());
      /* write record length at beginning of record */
      if (WRITE_UNF_LEN)
        UNF_ERR(__io_errno());
      if (adjust_fpos(Fcb, (seekoffx_t)bytecnt, SEEK_CUR) != 0)
        UNF_ERR(__io_errno());
    }
    /* If this record is a continuation of the previous record, set the
       continuation bit in this record's trailing length word. */
    if (continued) {
      if (!f90_old_huge_rec_fmt()) {
        unf_rec.u.s.bytecnt = -unf_rec.u.s.bytecnt;
      } else {
        unf_rec.u.s.bytecnt |= CONT_FLAG;
      }
    }
    continued = to_be_continued;
    /* write record length at end of record */
    if (WRITE_UNF_LEN)
      UNF_ERR(__io_errno());
  } else if (Fcb->reclen > unf_rec.u.s.bytecnt) {
    /*  pad record for direct-access file: */
    ret_err = __fortio_zeropad(Fcb->fp, Fcb->reclen - unf_rec.u.s.bytecnt);
    if (ret_err != 0)
      UNF_ERR(ret_err);
  }

  return 0;
}

static int
skip_to_nextrec(void)
{
  /* It's assumed that data has been read from the record */
  if (Fcb->acc != FIO_DIRECT) {
    if (rec_len == unf_rec.u.s.bytecnt) {
      /*  Tuning consideration:  it's better to skip 4 bytes by
          reading instead of using a relative fseek.  */
      if (adjust_fpos(Fcb, RCWSZ, SEEK_CUR))
        UNF_ERR(__io_errno());
    } else {
      Fcb->coherent = 0;
      if (adjust_fpos(Fcb, (seekoffx_t)(rec_len - unf_rec.u.s.bytecnt + RCWSZ),
                      SEEK_CUR))
        return (__io_errno());
    }
  } else if (unf_rec.u.s.bytecnt < rec_len) {
    Fcb->coherent = 0;
    if (__io_fseek(Fcb->fp, (seekoffx_t)(rec_len - unf_rec.u.s.bytecnt),
                   SEEK_CUR) != 0)
      return (__io_errno());
  }
  return 0;
}

__INT_T
ENTF90IO(UNF_END, unf_end)()

{
  int i, s = 0;
  int buffOffset;
  G *tmp_gbl;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_unf_end();

  /* recursive i/o and check all recursive fcb, starting from latest recursive
   */

  tmp_gbl = NULL;
  if (gbl_avl > 1) {
    for (i = gbl_avl - 2; i >= 0; --i) {
      if (gbl_head[i].Fcb == Fcb) {
        tmp_gbl = &gbl_head[i];
        break;
      }
    }
  }
  if (tmp_gbl) {
    tmp_gbl->rw_size = rw_size;
    tmp_gbl->rec_in_buf = rec_in_buf;
    tmp_gbl->rec_len = rec_len;
    tmp_gbl->io_transfer = io_transfer;
    tmp_gbl->continued = continued;
    memcpy(&tmp_gbl->unf_rec, &unf_rec, sizeof(unf_rec_struct));
    buffOffset = buf_ptr - unf_rec.buf;
    tmp_gbl->buf_ptr = tmp_gbl->unf_rec.buf + buffOffset;
  }

  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}

/* --------------------------------------------------------------------- */
/* --------------unformatted i/o with byte swapping  ------------------- */
/* --------------------------------------------------------------------- */

/** \brief Initialize global flags to prepare for unformatted I/O, and if the
 * file isn't opened, open it (if possible).  */
int
__f90io_usw_init(__INT_T *read,   /* TRUE indicates READ statement. */
                 __INT_T *unit,   /* unit number. */
                 __INT_T *rec,    /* record number for direct access */
                 __INT_T *bitv,   /* same as for ENTF90IO(open). */
                 __INT_T *iostat) /* same as for ENTF90IO(open). */
{
  int s = 0;
  save_gbl();
  if (*read)
    __fortio_errinit(*unit, *bitv, iostat, "unformatted read");
  else
    __fortio_errinit(*unit, *bitv, iostat, "unformatted write");

  allocate_new_gbl();
  Fcb = __fortio_rwinit(*unit, FIO_UNFORMATTED, rec, 1 - *read);
  if (Fcb == NULL) {
    if (fioFcbTbls.eof)
      return EOF_FLAG;
    /* TBD - does there need to be fioFcbTbls.eor */
    return ERR_FLAG;
  }
  continued = FALSE;
  actual_init = TRUE;

  s = __unf_init(*read, !Fcb->native);
  actual_init = FALSE;
  return s;
}

__INT_T
ENTF90IO(USW_INIT, usw_init)
(__INT_T *read,   /* TRUE indicates READ statement. */
 __INT_T *unit,   /* unit number. */
 __INT_T *rec,    /* record number for direct access */
 __INT_T *bitv,   /* same as for ENTF90IO(open). */
 __INT_T *iostat) /* same as for ENTF90IO(open). */
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_usw_init(read, unit, rec, bitv, iostat);
  if (s != 0) {
    free_gbl();
    restore_gbl();
    __fortio_errend03();
  }
  return DIST_STATUS_BCST(s);
}

/* ----------------------------------------------------------------------- */

/** \brief  Read/copy data from an unformatted record file. */
int
__f90io_usw_read(int type,   /* Type of data */
                 long count, /* number of items of specified type
                              * to read.  May be <= 0 */
                 int stride, /* distance in bytes between items */
                 char *item, /* where to xfer data */
                 __CLEN_T item_length)
{
  size_t nbytes; /* number of bytes to read from current record */
  size_t resid;  /* number of bytes to read from following records */
  int offset;    /* offset into item */
  int ret_val;
  char *item_ptr = item;

  /* first check for errors: */
  if (fioFcbTbls.eof) {
    ret_val = EOF_FLAG;
    goto uswr_err;
  }
  if (fioFcbTbls.error || !Fcb) {
    ret_val = ERR_FLAG;
    goto uswr_err;
  }

  if (Fcb->native)
    return __f90io_unf_read(type, count, stride, item, item_length);

  assert(item_ptr != NULL);

  if (count <= 0) /* treat this call as no-op */
    return 0;
  io_transfer = TRUE;

  nbytes = (size_t)count * item_length;
  offset = 0;

usw_read_do_resid:
  resid = 0;
  if (!Fcb->binary) {
    if (unf_rec.u.s.bytecnt + nbytes > (size_t)rec_len) {
      /* Not enough data in the current record to satisfy the request.
         If data continues in the next record, get the residual there. */
      if (!continued) {
        (void)skip_to_nextrec();
        ret_val = __fortio_error(FIO_ETOOBIG);
        goto uswr_err;
      }
      resid = unf_rec.u.s.bytecnt + nbytes - rec_len;
      nbytes -= resid;
    }

    if (rec_len == 0)
      return 0;
  }

  /* read directly into item if possible  (consecutive items) */

  if ((__CLEN_T)stride == item_length) {
    if (__io_fread(item_ptr, nbytes, 1, Fcb->fp) != 1) {
      if (__io_feof(Fcb->fp))
        ret_val = __fortio_error(FIO_EEOF);
      else
        ret_val = __fortio_error(__io_errno());
      goto uswr_err;
    }
    unf_rec.u.s.bytecnt += nbytes;
    if (resid > 0) {
      /* There is more data in the next record. */
      if ((ret_val = __usw_end(TO_BE_CONTINUED)) != 0) {
        goto uswr_err;
      }
      if ((ret_val = __unf_init(read_flag, !Fcb->native)) != 0) {
        goto uswr_err;
      }
      item_ptr += nbytes;
      nbytes = resid;
      io_transfer = TRUE;
      goto usw_read_do_resid;
    }
    __fortio_swap_bytes(item, type, count);
    return 0;
  }

  if (Fcb->asy_rw) { /* stop any async i/o */
    Fcb->asy_rw = 0;
    if (Fio_asy_disable(Fcb->asyptr) == -1) {
      ret_val = __fortio_error(__io_errno());
      goto uswr_err;
    }
  }

  /* copy 'count' items from stream into 'item', skipping by 'stride' */

  while (nbytes > 0) {
    int read_length;

    /* Read the lesser of
            bytes remaining in record (nbytes), or
            bytes needed to fill the item (item_length - offset) */
    read_length =
        (nbytes < item_length - offset ? nbytes : item_length - offset);
    if (__io_fread(item + offset, read_length, 1, Fcb->fp) != 1) {
      if (__io_feof(Fcb->fp))
        ret_val = __fortio_error(FIO_EEOF);
      else
        ret_val = __fortio_error(__io_errno());
      goto uswr_err;
    }
    unf_rec.u.s.bytecnt += read_length;
    nbytes -= read_length;
    offset += read_length;
    if ((__CLEN_T)offset == item_length) {
      __fortio_swap_bytes(item_ptr, type, 1);
      item += stride;
      offset = 0;
    }
  }
  if (resid > 0) {
    /* There is more data in the next record. */
    if ((ret_val = __usw_end(TO_BE_CONTINUED)) != 0) {
      goto uswr_err;
    }
    if ((ret_val = __unf_init(read_flag, !Fcb->native)) != 0) {
      goto uswr_err;
    }
    nbytes = resid;
    io_transfer = TRUE;
    goto usw_read_do_resid;
  }

  return 0;
uswr_err:
  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return ret_val;
}

__INT_T
ENTF90IO(USW_READA, usw_reada)
(__INT_T *type,   /* Type of data */
 __INT_T *count,  /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN64(item))   /* length for character item */
{
  int s = 0;
  int ioproc;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_usw_read(*type, *count, *stride, CADR(item), len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, CADR(item), *count, *stride, *type, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(USW_READ, usw_read)
(__INT_T *type,   /* Type of data */
 __INT_T *count,  /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN(item))     /* length for character item */
{
  return ENTF90IO(USW_READA, usw_reada)(type, count, stride, CADR(item),
                                        (__CLEN_T)CLEN(item));
}

/* same as usw_read, but item may be array - for usw_read, the compiler
 * scalarizes.
 */
__INT_T
ENTF90IO(USW_READ_AA, usw_read_aa)
(__INT_T *type,   /* Type of data */
 __INT_T *count,  /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN64(item))   /* length for character item */
{
  int s = 0;
  int ioproc;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_usw_read(*type, *count, *stride, CADR(item), len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, CADR(item), *count, *stride, *type, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(USW_READ_A, usw_read_a)
(__INT_T *type,   /* Type of data */
 __INT_T *count,  /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN(item))     /* length for character item */
{
  return ENTF90IO(USW_READ_AA, usw_read_aa)(type, count, stride, CADR(item),
                                            (__CLEN_T)CLEN(item));
}

__INT_T
ENTF90IO(USW_READ64_AA, usw_read64_aa)
(__INT_T *type,   /* Type of data */
 __INT8_T *count, /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN64(item))   /* length for character item */
{
  int s = 0;
  int ioproc;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  ioproc = GET_DIST_IOPROC;
  if (LOCAL_MODE || GET_DIST_LCPU == ioproc)
    s = __f90io_usw_read(*type, *count, *stride, CADR(item), len);
  if (!LOCAL_MODE)
    DIST_RBCSTL(ioproc, CADR(item), *count, *stride, *type, len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(USW_READ64_A, usw_read64_a)
(__INT_T *type,   /* Type of data */
 __INT8_T *count, /* number of items of specified type
                   * to read.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items */
 DCHAR(item)      /* where to xfer data */
 DCLEN(item))     /* length for character item */
{
  return ENTF90IO(USW_READ64_AA, usw_read64_aa)(type, count, stride, CADR(item),
                                                (__CLEN_T)CLEN(item));
}

/* ----------------------------------------------------------------- */

int
__f90io_usw_write(int type,   /* data type of data (see above). */
                  long count, /* number of items of specified type
                               * to write.  May be <= 0 */
                  int stride, /* distance in bytes between items. */
                  char *item, /* where to get data from */
                  __CLEN_T item_length)
{
  long i;        /* loop index */
  __CLEN_T j;    /* loop index */
  size_t nbytes; /* # of bytes to write for this call */
  int bs_tmp;
  int ret_val;

  /* first check for errors: */
  assert(fioFcbTbls.eof == 0);
  if (fioFcbTbls.error || !Fcb) {
    ret_val = ERR_FLAG;
    goto unf_write_err;
  }

  if (Fcb->native)
    return __f90io_unf_write(type, count, stride, item, item_length);

  assert(item != NULL);

  if (count <= 0)
    return 0;
  io_transfer = TRUE;

  nbytes = (size_t)count * item_length;

  if (Fcb->acc == FIO_DIRECT &&
      unf_rec.u.s.bytecnt + nbytes > (size_t)rec_len) {
    ret_val = __fortio_error(FIO_ETOOBIG);
    goto unf_write_err;
  }

  if (item_length == (__CLEN_T)stride) {
    /*  optimize if we have consecutive items  */
    size_t resid;

    if (unf_rec.u.s.bytecnt + nbytes > MAX_REC_SIZE) {
      resid = unf_rec.u.s.bytecnt + nbytes -
              ((MAX_REC_SIZE / item_length) * item_length);
      nbytes -= resid;
    } else
      resid = 0;
    if (rw_size + nbytes > IOBUFSIZE || resid > 0) {
      if (DBGBIT(0x4))
        __io_printf("unit stride flush, rw_size=%" GBL_SIZE_T_FORMAT
                    ", in_buf:%d\n",
                    rw_size, rec_in_buf);
      if (rec_in_buf) {
        if (!Fcb->binary) {
          bs_tmp = unf_rec.u.s.bytecnt + nbytes;
          __fortio_swap_bytes((char *)&bs_tmp, __INT, 1);
          if ((FWRITE(&bs_tmp, RCWSZ, 1, Fcb->fp)) != 1) {
            ret_val = __fortio_error(__io_errno());
            goto unf_write_err;
          }
        }
        if (WRITE_UNF_BUF) {
          ret_val = __fortio_error(__io_errno());
          goto unf_write_err;
        }
        rec_in_buf = FALSE;
        unf_rec.u.s.bcnt = unf_rec.u.s.bytecnt + nbytes;
      } else {
        if (WRITE_UNF_BUF) {
          ret_val = __fortio_error(__io_errno());
          goto unf_write_err;
        }
      }
      if (DBGBIT(0x4))
        __io_printf("to nonunit stride copy, nbytes=%" GBL_SIZE_T_FORMAT "\n",
                    nbytes);
      rw_size = 0;
      buf_ptr = unf_rec.buf;
      goto nonunit_cp;
    }
    if (DBGBIT(0x4))
      __io_printf("unit stride copy, nbytes=%" GBL_SIZE_T_FORMAT
                  ", rw_size=%" GBL_SIZE_T_FORMAT ", in_buf:%d\n",
                  nbytes, rw_size, rec_in_buf);
    (void)memcpy(buf_ptr, item, nbytes);
    __fortio_swap_bytes(buf_ptr, type, count);
    unf_rec.u.s.bytecnt += nbytes;
    buf_ptr += nbytes;
    rw_size += nbytes;
    return 0;
  }

  /* copy 'count' items from 'item' into buffer, skipping by stride  */

nonunit_cp:
  for (i = 0; i < count; i++, item += (stride - item_length)) {
    int rec_full;

    if ((unf_rec.u.s.bytecnt + item_length) > MAX_REC_SIZE)
      rec_full = TRUE;
    else {
      rec_full = FALSE;
      unf_rec.u.s.bytecnt += item_length;
    }
    if ((rw_size + item_length) >= IOBUFSIZE || rec_full) {
      if (DBGBIT(0x4))
        __io_printf("non-unit stride flush, nbytes=%" GBL_SIZE_T_FORMAT
                    ", in_buf:%d\n",
                    rw_size, rec_in_buf);
      if (rec_in_buf) {
        if (!Fcb->binary) {
          bs_tmp = unf_rec.u.s.bytecnt;
          __fortio_swap_bytes((char *)&bs_tmp, __INT, 1);
          if ((FWRITE(&bs_tmp, RCWSZ, 1, Fcb->fp)) != 1) {
            ret_val = __fortio_error(__io_errno());
            goto unf_write_err;
          }
        }
        if (WRITE_UNF_BUF) {
          ret_val = __fortio_error(__io_errno());
          goto unf_write_err;
        }
        rec_in_buf = FALSE;
        unf_rec.u.s.bcnt = unf_rec.u.s.bytecnt;
      } else {
        if (WRITE_UNF_BUF) {
          ret_val = __fortio_error(__io_errno());
          goto unf_write_err;
        }
      }
      rw_size = 0;
      buf_ptr = unf_rec.buf;
      if (rec_full) {
        /* Start a new record. */
        if ((ret_val = __usw_end(TO_BE_CONTINUED)) != 0)
          return ret_val;
        if ((ret_val = __unf_init(read_flag, Fcb->byte_swap)) != 0)
          return ret_val;
        /* Current item was not written; try again. */
        i--;
        item -= (stride - item_length);
        io_transfer = TRUE;
        continue;
      }
      if (item_length > IOBUFSIZE) {
        /* only occurs if the type is __STR or __NCHAR */
        if (type == __STR) {
          if ((unf_fwrite(item, item_length, 1, Fcb)) != TRUE) {
            ret_val = __fortio_error(__io_errno());
            goto unf_write_err;
          }
        } else {
          char *pp;
          pp = malloc(item_length);
          if (pp == NULL)
            return __fortio_error(FIO_ENOMEM);
          (void)memcpy(pp, item, item_length);
          __fortio_swap_bytes(pp, type, item_length >> 1);
          if ((FWRITE(pp, item_length, 1, Fcb->fp)) != 1) {
            ret_val = __fortio_error(__io_errno());
            goto unf_write_err;
          }
          free(pp);
        }
        item += item_length;
        continue;
      }
    }
    /* ??? one copy per byte.  Maybe optimize later */
    for (j = 0; j < item_length; j++)
      *buf_ptr++ = *item++;

    __fortio_swap_bytes(buf_ptr - item_length, type, 1);
    rw_size += item_length;
  }

  return 0;

unf_write_err:
  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return ret_val;
}

__INT_T
ENTF90IO(USW_WRITEA, usw_writea)
(__INT_T *type,   /* data type of data (see above). */
 __INT_T *count,  /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN64(item))   /* length for character item */
{
  int s = 0;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_usw_write(*type, *count, *stride, CADR(item), len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(USW_WRITE, usw_write)
(__INT_T *type,   /* data type of data (see above). */
 __INT_T *count,  /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN(item))     /* length for character item */
{
  return ENTF90IO(USW_WRITEA, usw_writea)(type, count, stride, CADR(item),
                                          (__CLEN_T)CLEN(item));
}

/** \brief same as usw_write, but item may be array - for usw_write, the
 * compiler scalarizes.
 */
__INT_T
ENTF90IO(USW_WRITE_AA, usw_write_aa)
(__INT_T *type,   /* data type of data (see above). */
 __INT_T *count,  /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN64(item))   /* length for character item */
{
  int s = 0;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_usw_write(*type, *count, *stride, CADR(item), len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(USW_WRITE_A, usw_write_a)
(__INT_T *type,   /* data type of data (see above). */
 __INT_T *count,  /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN(item))     /* length for character item */
{
  return ENTF90IO(USW_WRITE_AA, usw_write_aa)(type, count, stride, CADR(item),
                                              (__CLEN_T)CLEN(item));
}

/** \brief same as usw_write, but item may be array - for usw_write, the
 * compiler scalarizes.
 */
__INT_T
ENTF90IO(USW_WRITE64_AA, usw_write64_aa)
(__INT_T *type,   /* data type of data (see above). */
 __INT8_T *count, /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN64(item))   /* length for character item */
{
  int s = 0;
  __CLEN_T len;

  if (*type == __STR)
    len = CLEN(item);
  else
    len = GET_DIST_SIZE_OF(*type);

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_usw_write(*type, *count, *stride, CADR(item), len);
  return DIST_STATUS_BCST(s);
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(USW_WRITE64_A, usw_write64_a)
(__INT_T *type,   /* data type of data (see above). */
 __INT8_T *count, /* number of items of specified type
                   * to write.  May be <= 0 */
 __INT_T *stride, /* distance in bytes between items. */
 DCHAR(item)      /* where to get data from */
 DCLEN(item))     /* length for character item */
{
  return ENTF90IO(USW_WRITE64_AA, usw_write64_aa)(
      type, count, stride, CADR(item), (__CLEN_T)CLEN(item));
}
/* -------------------------------------------------------------------- */

/** \ brief Finish up unformatted read or write.  If current I/O is a read,
 * write the current buffer to the file.  Whether a read or a write,
 * free the buffer. */
int
__f90io_usw_end(void)
{
  if (fioFcbTbls.error)
    return ERR_FLAG;
  if (fioFcbTbls.eof || !Fcb)
    return EOF_FLAG;
  if (has_same_fcb)
    return 0;

  if (Fcb->native)
    return __f90io_unf_end();

  return __usw_end(!TO_BE_CONTINUED);
}

static int
__usw_end(bool to_be_continued)
{
  int ret_err;
  int bs_tmp;

  /* if read and direct (variable length) then seek past the trailing
   * 4-byte integer that indicates the record's size so the next read/write
   * will be at the next record: */

  if (read_flag) {
    if (Fcb->binary) {
      Fcb->coherent = 0;
      return 0;
    }
    if (!io_transfer) {
      /*
       * read with no input list-- seek past current record.
       * this only happens if we have a READ statement without any
       * items.
       */
      if (Fcb->acc != FIO_DIRECT)
        ret_err = __io_fseek(Fcb->fp, (seekoffx_t)rec_len + RCWSZ, SEEK_CUR);
      else
        ret_err = __io_fseek(Fcb->fp, (seekoffx_t)rec_len, SEEK_CUR);
      if (ret_err)
        UNF_ERR(__io_errno());
      Fcb->coherent = 0;
      return 0;
    }

    ret_err = skip_to_nextrec();
    if (ret_err)
      UNF_ERR(ret_err);

    /* If we want to use data continued into the next record, we're done.
       Else we need to seek to the actual end of data by watching the
       continue flags. */
    if (to_be_continued)
      return 0;
    while (continued) {
      if (__io_fread(&rec_len, 4, 1, Fcb->fp) != 1)
        UNF_ERR(__io_errno());
      __fortio_swap_bytes((char *)&rec_len, __INT, 1);
      if (!f90_old_huge_rec_fmt()) {
        if (__io_fseek(Fcb->fp, -rec_len + 4, SEEK_CUR)) {
          UNF_ERR(__io_errno());
        }
        continued = (rec_len < 0);
      } else {
        if (__io_fseek(Fcb->fp, (rec_len &= ~CONT_FLAG) + 4, SEEK_CUR)) {
          UNF_ERR(__io_errno());
        }
        continued = (rec_len & CONT_FLAG);
      }
    }
    return 0;
  }

  /*
   * Case 2:  Write statment.  write the buffer to the file.  check if direct
   * (variable length), and if so poke in the reclen to the first and
   * last 4 bytes of the buffer else pad the rest of the buffer with
   * zeroes
   */
  if (!io_transfer) {
    if (Fcb->acc != FIO_DIRECT) { /* write 0 length record */
      if (Fcb->binary)
        return 0;
      ret_err = __fortio_zeropad(Fcb->fp, RCWSZ << 1);
      if (ret_err != 0)
        UNF_ERR(ret_err);
      return 0;
    }
    rw_size = 0;
  }

  if (rec_in_buf) {
    if (Fcb->binary) {
      if (WRITE_UNF_BUF)
        UNF_ERR(__io_errno());
      return 0;
    }
    bs_tmp = unf_rec.u.s.bytecnt;
    __fortio_swap_bytes((char *)&bs_tmp, __INT, 1);
    if ((FWRITE(&bs_tmp, RCWSZ, 1, Fcb->fp)) != 1)
      UNF_ERR(__io_errno());
    if (WRITE_UNF_BUF)
      UNF_ERR(__io_errno());
    if ((FWRITE(&bs_tmp, RCWSZ, 1, Fcb->fp)) != 1)
      UNF_ERR(__io_errno());
    return 0;
  }

  if (WRITE_UNF_BUF)
    UNF_ERR(__io_errno());
  rw_size = 0;
  buf_ptr = unf_rec.buf;

  if (Fcb->acc != FIO_DIRECT) {
    if (Fcb->binary)
      return 0;
    /* If this record's data is to be continued in the next record,
       set the continuation bit in this record's leading length word. */
    bs_tmp = to_be_continued ? -unf_rec.u.s.bytecnt : unf_rec.u.s.bytecnt;
    __fortio_swap_bytes((char *)&bs_tmp, __INT, 1);
    if (unf_rec.u.s.bcnt != unf_rec.u.s.bytecnt || to_be_continued) {
      /* seek to record's beginning length word */
      if (__io_fseek(Fcb->fp,
                     (seekoffx_t)(-unf_rec.u.s.bytecnt) - (seekoffx_t)(RCWSZ),
                     SEEK_CUR) != 0)
        UNF_ERR(__io_errno());
      /* write record length at beginning of record */
      if ((FWRITE(&bs_tmp, RCWSZ, 1, Fcb->fp)) != 1)
        UNF_ERR(__io_errno());
      if (__io_fseek(Fcb->fp, (seekoffx_t)unf_rec.u.s.bytecnt, SEEK_CUR) != 0)
        UNF_ERR(__io_errno());
      if (to_be_continued && !continued) {
        bs_tmp = f90_old_huge_rec_fmt() ? -bs_tmp : bs_tmp & ~CONT_FLAG_SW;
      }
    }
    /* If this record is a continuation of the previous record, set the
       continuation bit in this record's trailing length word. */
    if (continued)
      bs_tmp = f90_old_huge_rec_fmt() ? bs_tmp : (bs_tmp | CONT_FLAG_SW);
    continued = to_be_continued;
    /* write record length at end of record */
    if ((FWRITE(&bs_tmp, RCWSZ, 1, Fcb->fp)) != 1)
      UNF_ERR(__io_errno());
  } else if (Fcb->reclen > unf_rec.u.s.bytecnt) {
    /*  pad record for direct-access file: */
    ret_err = __fortio_zeropad(Fcb->fp, Fcb->reclen - unf_rec.u.s.bytecnt);
    if (ret_err != 0)
      UNF_ERR(ret_err);
  }

  return 0;
}

__INT_T
ENTF90IO(USW_END, usw_end)()
{
  int i, s = 0;
  int buffOffset;
  G *tmp_gbl;

  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = __f90io_usw_end();

  /* recursive i/o and check all recursive fcb, starting from latest recursive
   */

  tmp_gbl = NULL;
  if (gbl_avl > 1) {
    for (i = gbl_avl - 2; i >= 0; --i) {
      if (gbl_head[i].Fcb == Fcb) {
        tmp_gbl = &gbl_head[i];
        break;
      }
    }
  }
  if (tmp_gbl) {
    tmp_gbl->rw_size = rw_size;
    tmp_gbl->rec_in_buf = rec_in_buf;
    tmp_gbl->rec_len = rec_len;
    tmp_gbl->io_transfer = io_transfer;
    tmp_gbl->continued = continued;
    memcpy(&tmp_gbl->unf_rec, &unf_rec, sizeof(unf_rec_struct));
    buffOffset = buf_ptr - unf_rec.buf;
    tmp_gbl->buf_ptr = tmp_gbl->unf_rec.buf + buffOffset;
  }

  free_gbl();
  restore_gbl();
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}

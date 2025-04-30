#ifndef __A_OUT_GNU_H__
#define __A_OUT_GNU_H__

#include <bits/a.out.h>

#define __GNU_EXEC_MACROS__

/*
 * OSF/1 ECOFF header structs.  ECOFF files consist of:
 *      - a file header (struct filehdr),
 *      - an a.out header (struct aouthdr),
 *      - one or more section headers (struct scnhdr).
 *        The filhdr's "f_nscns" field contains the
 *        number of section headers.
 */

struct filehdr
{
  /* OSF/1 "file" header */
  unsigned short f_magic, f_nscns;
  unsigned int   f_timdat;
  unsigned long  f_symptr;
  unsigned int   f_nsyms;
  unsigned short f_opthdr, f_flags;
};

struct aouthdr
{
  unsigned long info;		/* After that it looks quite normal..  */
  unsigned long tsize;
  unsigned long dsize;
  unsigned long bsize;
  unsigned long entry;
  unsigned long text_start;	/* With a few additions that actually make sense.  */
  unsigned long data_start;
  unsigned long bss_start;
  unsigned int  gprmask, fprmask; /* Bitmask of general & floating point regs used in binary.  */
  unsigned long gpvalue;
};

struct scnhdr
{
  char           s_name[8];
  unsigned long  s_paddr;
  unsigned long  s_vaddr;
  unsigned long  s_size;
  unsigned long  s_scnptr;
  unsigned long  s_relptr;
  unsigned long  s_lnnoptr;
  unsigned short s_nreloc;
  unsigned short s_nlnno;
  unsigned int   s_flags;
};

struct exec
{
  /* OSF/1 "file" header */
  struct filehdr fh;
  struct aouthdr ah;
};

#define a_info		ah.info
#define a_text		ah.tsize
#define a_data		ah.dsize
#define a_bss		ah.bsize
#define a_entry		ah.entry
#define a_textstart	ah.text_start
#define a_datastart	ah.data_start
#define a_bssstart	ah.bss_start
#define a_gprmask	ah.gprmask
#define a_fprmask	ah.fprmask
#define a_gpvalue	ah.gpvalue


#define AOUTHSZ		sizeof (struct aouthdr)
#define SCNHSZ		sizeof (struct scnhdr)
#define SCNROUND	16

enum machine_type
{
  M_OLDSUN2 = 0,
  M_68010 = 1,
  M_68020 = 2,
  M_SPARC = 3,
  M_386 = 100,
  M_MIPS1 = 151,
  M_MIPS2 = 152
};

#define N_MAGIC(exec)	((exec).a_info & 0xffff)
#define N_MACHTYPE(exec) ((enum machine_type)(((exec).a_info >> 16) & 0xff))
#define N_FLAGS(exec)	(((exec).a_info >> 24) & 0xff)
#define N_SET_INFO(exec, magic, type, flags) \
  ((exec).a_info = ((magic) & 0xffff)					\
   | (((int)(type) & 0xff) << 16)					\
   | (((flags) & 0xff) << 24))
#define N_SET_MAGIC(exec, magic) \
  ((exec).a_info = ((exec).a_info & 0xffff0000) | ((magic) & 0xffff))
#define N_SET_MACHTYPE(exec, machtype) \
  ((exec).a_info =							\
   ((exec).a_info&0xff00ffff) | ((((int)(machtype))&0xff) << 16))
#define N_SET_FLAGS(exec, flags) \
  ((exec).a_info =							\
   ((exec).a_info&0x00ffffff) | (((flags) & 0xff) << 24))

/* Code indicating object file or impure executable.  */
#define OMAGIC 0407
/* Code indicating pure executable.  */
#define NMAGIC 0410
/* Code indicating demand-paged executable.  */
#define ZMAGIC 0413
/* This indicates a demand-paged executable with the header in the text.
   The first page is unmapped to help trap NULL pointer references.  */
#define QMAGIC 0314
/* Code indicating core file.  */
#define CMAGIC 0421

#define N_TRSIZE(x)	0
#define N_DRSIZE(x)	0
#define N_SYMSIZE(x)	0
#define N_BADMAG(x) \
  (N_MAGIC(x) != OMAGIC	&& N_MAGIC(x) != NMAGIC				\
   && N_MAGIC(x) != ZMAGIC && N_MAGIC(x) != QMAGIC)
#define _N_HDROFF(x)	(1024 - sizeof (struct exec))
#define N_TXTOFF(x) \
  ((long) N_MAGIC(x) == ZMAGIC ? 0					\
   : ((sizeof (struct exec) + (x).fh.f_nscns * SCNHSZ + SCNROUND - 1)	\
      & ~(SCNROUND - 1)))

#define N_DATOFF(x)	(N_TXTOFF(x) + (x).a_text)
#define N_TRELOFF(x)	(N_DATOFF(x) + (x).a_data)
#define N_DRELOFF(x)	(N_TRELOFF(x) + N_TRSIZE(x))
#define N_SYMOFF(x)	(N_DRELOFF(x) + N_DRSIZE(x))
#define N_STROFF(x)	(N_SYMOFF(x) + N_SYMSIZE(x))

/* Address of text segment in memory after it is loaded.  */
#define N_TXTADDR(x)	((x).a_textstart)

/* Address of data segment in memory after it is loaded.  */
#define SEGMENT_SIZE	1024

#define _N_SEGMENT_ROUND(x) (((x) + SEGMENT_SIZE - 1) & ~(SEGMENT_SIZE - 1))
#define _N_TXTENDADDR(x) (N_TXTADDR(x)+(x).a_text)

#define N_DATADDR(x)	((x).a_datastart)
#define N_BSSADDR(x)	((x).a_bssstart)

#if !defined (N_NLIST_DECLARED)
struct nlist
{
  union
    {
      char *n_name;
      struct nlist *n_next;
      long n_strx;
    } n_un;
  unsigned char n_type;
  char n_other;
  short n_desc;
  unsigned long n_value;
};
#endif /* no N_NLIST_DECLARED.  */

#define N_UNDF	0
#define N_ABS	2
#define N_TEXT	4
#define N_DATA	6
#define N_BSS	8
#define N_FN	15
#define N_EXT	1
#define N_TYPE	036
#define N_STAB	0340
#define N_INDR	0xa
#define	N_SETA	0x14	/* Absolute set element symbol.  */
#define	N_SETT	0x16	/* Text set element symbol.  */
#define	N_SETD	0x18	/* Data set element symbol.  */
#define	N_SETB	0x1A	/* Bss set element symbol.  */
#define N_SETV	0x1C	/* Pointer to set vector in data area.  */

#if !defined (N_RELOCATION_INFO_DECLARED)
/* This structure describes a single relocation to be performed.
   The text-relocation section of the file is a vector of these structures,
   all of which apply to the text section.
   Likewise, the data-relocation section applies to the data section.  */

struct relocation_info
{
  int r_address;
  unsigned int r_symbolnum:24;
  unsigned int r_pcrel:1;
  unsigned int r_length:2;
  unsigned int r_extern:1;
  unsigned int r_pad:4;
};
#endif /* no N_RELOCATION_INFO_DECLARED.  */

#endif /* __A_OUT_GNU_H__ */

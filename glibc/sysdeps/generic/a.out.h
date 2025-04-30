#ifndef __A_OUT_GNU_H__
#define __A_OUT_GNU_H__

#ifdef	__cplusplus
extern "C" {
#endif

#define __GNU_EXEC_MACROS__

#ifndef __STRUCT_EXEC_OVERRIDE__

struct exec
{
  unsigned long a_info;		/* Use macros N_MAGIC, etc for access */
  unsigned a_text;		/* size of text, in bytes */
  unsigned a_data;		/* size of data, in bytes */
  unsigned a_bss;		/* size of uninitialized data area, in bytes */
  unsigned a_syms;		/* length of symbol table data, in bytes */
  unsigned a_entry;		/* start address */
  unsigned a_trsize;		/* size of reloc info for text, in bytes */
  unsigned a_drsize;		/* size of reloc info for data, in bytes */

#if defined (sequent) && defined (i386)
  struct gdtbl
    {				/* Global Descriptor Table */
      unsigned g_code[2];
      unsigned g_data[2];
      unsigned g_desc[2];
    } a_gdtbl;
   unsigned a_shdata;		/* size of initialized shared data */
   unsigned a_shbss;		/* size of uninitialized shared data */
   unsigned a_shdrsize;		/* size of shared data relocation */
   unsigned a_bootstrap[11];	/* bootstrap for standalone */
   unsigned a_reserved[3];	/* reserved for future use */
   unsigned a_version;		/* object version */
#endif /* Sequent Symmetry, Dynix 3.x */
};

#endif /* __STRUCT_EXEC_OVERRIDE__ */

/* these go in the N_MACHTYPE field */
enum machine_type {
#if defined (M_OLDSUN2)
  M__OLDSUN2 = M_OLDSUN2,
#else
  M_OLDSUN2 = 0,
#endif
#if defined (M_68010)
  M__68010 = M_68010,
#else
  M_68010 = 1,
#endif
#if defined (M_68020)
  M__68020 = M_68020,
#else
  M_68020 = 2,
#endif
#if defined (M_SPARC)
  M__SPARC = M_SPARC,
#else
  M_SPARC = 3,
#endif
  /* skip a bunch so we don't run into any of sun's numbers */
  M_386 = 100
};

#if defined (sequent) && defined (i386)

/* Dynix 3 wants the magic number to be the whole first longword.  */

#define N_MAGIC(exec)		((exec).a_info)
#define N_MACHTYPE(exec)	0
#define N_FLAGS(exec)		0
#define N_SET_INFO(exec, magic, type, flags) N_SET_MAGIC ((exec), (magic))
#define N_SET_MAGIC(exec, magic)	((exec).a_info = (magic))
#define N_SET_MACHTYPE(exec, machtype)	((void) 0)
#define N_SET_FLAGS(exec, flags)	((void) 0)
#define	OMAGIC		0x12eb	/* impure format - for .o's */
#define	ZMAGIC		0x22eb	/* demand load format - zero at zero */
#define NMAGIC		you lose /* syntax error -- no pure format */
#define N_BADMAG(x)	(N_MAGIC(x) != OMAGIC && N_MAGIC(x) != ZMAGIC)
#define N_ADDRADJ(x)	(N_MAGIC(x) == ZMAGIC ? SEGMENT_SIZE : 0)
#define N_DATOFF(x)	(N_TXTOFF(x) + (x).a_text - N_ADDRADJ(x))
#define N_TRELOFF(x)	(N_DATOFF(x) + (x).a_data + (x).a_shdata)
#define N_SYMOFF(x)	(N_DRELOFF(x) + (x).a_drsize + (x).a_shdrsize)
#define N_TXTADDR(x)	SEGMENT_SIZE
#define N_COMM		0x0a	/** conflicts with N_INDR **/
#define N_FN		0x0c
/* Note that the Dynix binutils believe that N_SET[TDB] are
   N_SH{DATA,BSS,COMM} -- be wary when mixing GNU & Dynix objects.  */
#define PAGE_SIZE	4096
#define SEGMENT_SIZE	PAGE_SIZE

#else /* !(sequent && i386) */

#if !defined (N_MAGIC)
#define N_MAGIC(exec) ((exec).a_info & 0xffff)
#endif
#define N_MACHTYPE(exec) ((enum machine_type)(((exec).a_info >> 16) & 0xff))
#define N_FLAGS(exec) (((exec).a_info >> 24) & 0xff)
#define N_SET_INFO(exec, magic, type, flags) \
	((exec).a_info = ((magic) & 0xffff) \
	 | (((int)(type) & 0xff) << 16) \
	 | (((flags) & 0xff) << 24))
#define N_SET_MAGIC(exec, magic) \
	((exec).a_info = (((exec).a_info & 0xffff0000) | ((magic) & 0xffff)))

#define N_SET_MACHTYPE(exec, machtype) \
	((exec).a_info = \
	 ((exec).a_info&0xff00ffff) | ((((int)(machtype))&0xff) << 16))

#define N_SET_FLAGS(exec, flags) \
	((exec).a_info = \
	 ((exec).a_info&0x00ffffff) | (((flags) & 0xff) << 24))

#endif	/* sequent && i386 */

#ifndef OMAGIC
/* Code indicating object file or impure executable.  */
#define OMAGIC 0407
/* Code indicating pure executable.  */
#define NMAGIC 0410
/* Code indicating demand-paged executable.  */
#define ZMAGIC 0413
#endif /* not OMAGIC */

#if !defined (N_BADMAG)
#define N_BADMAG(x)					\
 (N_MAGIC(x) != OMAGIC && N_MAGIC(x) != NMAGIC		\
  && N_MAGIC(x) != ZMAGIC)
#endif

#define _N_BADMAG(x)					\
 (N_MAGIC(x) != OMAGIC && N_MAGIC(x) != NMAGIC		\
  && N_MAGIC(x) != ZMAGIC)

#ifndef	sparc
#define _N_HDROFF(x) (SEGMENT_SIZE - sizeof (struct exec))
#else
#define _N_HDROFF(x) (- sizeof (struct exec))
#endif

#if !defined (N_TXTOFF)
#define N_TXTOFF(x) \
 (N_MAGIC(x) == ZMAGIC ? _N_HDROFF((x)) + sizeof (struct exec) : sizeof (struct exec))
#endif

#if !defined (N_DATOFF)
#define N_DATOFF(x) (N_TXTOFF(x) + (x).a_text)
#endif

#if !defined (N_TRELOFF)
#define N_TRELOFF(x) (N_DATOFF(x) + (x).a_data)
#endif

#if !defined (N_DRELOFF)
#define N_DRELOFF(x) (N_TRELOFF(x) + (x).a_trsize)
#endif

#if !defined (N_SYMOFF)
#define N_SYMOFF(x) (N_DRELOFF(x) + (x).a_drsize)
#endif

#if !defined (N_STROFF)
#define N_STROFF(x) (N_SYMOFF(x) + (x).a_syms)
#endif

/* Address of text segment in memory after it is loaded.  */
#if !defined (N_TXTADDR)
#define N_TXTADDR(x) 0
#endif

/* Address of data segment in memory after it is loaded.
   Note that it is up to you to define SEGMENT_SIZE
   on machines not listed here.  */
#if	defined (hp300) || defined (mips)
#define	PAGE_SIZE	4096
#endif
#if	defined (sparc) || defined (NeXT)
#define	PAGE_SIZE	0x2000
#endif
#if	defined (sony) || (defined (sun) && defined (mc68000))
#define	SEGMENT_SIZE	0x2000
#endif	/* Sony or 68k Sun.  */
#ifdef is68k
#define SEGMENT_SIZE 0x20000
#endif
#if defined(m68k) && defined(PORTAR)
#define PAGE_SIZE 0x400
#endif
#if defined(i386) && !defined(sequent)
/* For COFF encapsulation.  */
#define SEGMENT_SIZE 0x400000
#endif
#ifndef	SEGMENT_SIZE
/* This used to be first in this paragraph and under:
   if (defined(vax) || defined(hp300) || defined(pyr) || defined(sparc) \
     || (defined(m68k) && defined(PORTAR)) \
     || defined (NeXT) || defined (mips)) */
#define SEGMENT_SIZE PAGE_SIZE
#endif
#ifndef	PAGE_SIZE
/* This value is for i386-minix, but that has no predefine.
   Making it default will only cause confusion on machines
   which have no proper value defined.  */
#define	PAGE_SIZE 16
#endif

#define	PAGSIZ	PAGE_SIZE
#define	SEGSIZ	SEGMENT_SIZE

#define _N_SEGMENT_ROUND(x) (((x) + SEGMENT_SIZE - 1) & ~(SEGMENT_SIZE - 1))

#define _N_TXTENDADDR(x) (N_TXTADDR(x)+(x).a_text)

#ifndef N_DATADDR
#define N_DATADDR(x) \
    (N_MAGIC(x)==OMAGIC? (_N_TXTENDADDR(x)) \
     : (_N_SEGMENT_ROUND (_N_TXTENDADDR(x))))
#endif

/* Address of bss segment in memory after it is loaded.  */
#if !defined (N_BSSADDR)
#define N_BSSADDR(x) (N_DATADDR(x) + (x).a_data)
#endif

#if !defined (N_NLIST_DECLARED)
struct nlist {
  union {
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

#if !defined (N_UNDF)
#define N_UNDF 0
#endif
#if !defined (N_ABS)
#define N_ABS 2
#endif
#if !defined (N_TEXT)
#define N_TEXT 4
#endif
#if !defined (N_DATA)
#define N_DATA 6
#endif
#if !defined (N_BSS)
#define N_BSS 8
#endif
#if !defined (N_COMM)
#define N_COMM 18
#endif
#if !defined (N_FN)
#define N_FN 15
#endif

#if !defined (N_EXT)
#define N_EXT 1
#endif
#if !defined (N_TYPE)
#define N_TYPE 036
#endif
#if !defined (N_STAB)
#define N_STAB 0340
#endif

/* The following type indicates the definition of a symbol as being
   an indirect reference to another symbol.  The other symbol
   appears as an undefined reference, immediately following this symbol.

   Indirection is asymmetrical.  The other symbol's value will be used
   to satisfy requests for the indirect symbol, but not vice versa.
   If the other symbol does not have a definition, libraries will
   be searched to find a definition.  */
#define N_INDR 0xa

/* The following symbols refer to set elements.
   All the N_SET[ATDB] symbols with the same name form one set.
   Space is allocated for the set in the text section, and each set
   element's value is stored into one word of the space.
   The first word of the space is the length of the set (number of elements).

   The address of the set is made into an N_SETV symbol
   whose name is the same as the name of the set.
   This symbol acts like a N_DATA global symbol
   in that it can satisfy undefined external references.  */

/* These appear as input to LD, in a .o file.  */
#define	N_SETA	0x14		/* Absolute set element symbol */
#define	N_SETT	0x16		/* Text set element symbol */
#define	N_SETD	0x18		/* Data set element symbol */
#define	N_SETB	0x1A		/* Bss set element symbol */

/* This is output from LD.  */
#define N_SETV	0x1C		/* Pointer to set vector in data area.  */

#if !defined (N_RELOCATION_INFO_DECLARED)
/* This structure describes a single relocation to be performed.
   The text-relocation section of the file is a vector of these structures,
   all of which apply to the text section.
   Likewise, the data-relocation section applies to the data section.  */

struct relocation_info
{
  /* Address (within segment) to be relocated.  */
  int r_address;
  /* The meaning of r_symbolnum depends on r_extern.  */
  unsigned int r_symbolnum:24;
  /* Nonzero means value is a pc-relative offset
     and it should be relocated for changes in its own address
     as well as for changes in the symbol or section specified.  */
  unsigned int r_pcrel:1;
  /* Length (as exponent of 2) of the field to be relocated.
     Thus, a value of 2 indicates 1<<2 bytes.  */
  unsigned int r_length:2;
  /* 1 => relocate with value of symbol.
          r_symbolnum is the index of the symbol
	  in file's the symbol table.
     0 => relocate with the address of a segment.
          r_symbolnum is N_TEXT, N_DATA, N_BSS or N_ABS
	  (the N_EXT bit may be set also, but signifies nothing).  */
  unsigned int r_extern:1;
  /* Four bits that aren't used, but when writing an object file
     it is desirable to clear them.  */
  unsigned int r_pad:4;
};
#endif /* no N_RELOCATION_INFO_DECLARED.  */

#ifdef	__cplusplus
}
#endif

#endif /* __A_OUT_GNU_H__ */

/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* I/O access is restricted to ISA port space (ports 0..65535).
   Modern devices hopefully are sane enough not to put any performance
   critical registers in i/o space.

   On the first call to ioperm, the entire (E)ISA port space is mapped
   into the virtual address space at address io.base.  mprotect calls
   are then used to enable/disable access to ports.  Per page, there
   are PAGE_SIZE>>IO_SHIFT I/O ports (e.g., 256 ports on a Low Cost Alpha
   based system using 8KB pages).

   Keep in mind that this code should be able to run in a 32bit address
   space.  It is therefore unreasonable to expect mmap'ing the entire
   sparse address space would work (e.g., the Low Cost Alpha chip has an
   I/O address space that's 512MB large!).  */

/* Make sure the ldbu/stb asms below are not expaneded to macros.  */
#ifndef __alpha_bwx__
asm(".arch ev56");
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/io.h>

#include <sysdep.h>
#include <sys/syscall.h>

#define PATH_ALPHA_SYSTYPE	"/etc/alpha_systype"
#define PATH_CPUINFO		"/proc/cpuinfo"

#define MAX_PORT	0x10000
#define vip		volatile int *
#define vuip		volatile unsigned int *
#define vusp		volatile unsigned short *
#define vucp		volatile unsigned char *

#define JENSEN_IO_BASE		(0x300000000UL)
#define JENSEN_SPARSE_MEM	(0x200000000UL)

/* With respect to the I/O architecture, APECS and LCA are identical,
   so the following defines apply to LCA as well.  */
#define APECS_IO_BASE		(0x1c0000000UL)
#define APECS_SPARSE_MEM	(0x200000000UL)
#define APECS_DENSE_MEM		(0x300000000UL)

/* The same holds for CIA and PYXIS, except for PYXIS we prefer BWX.  */
#define CIA_IO_BASE		(0x8580000000UL)
#define CIA_SPARSE_MEM		(0x8000000000UL)
#define CIA_DENSE_MEM		(0x8600000000UL)

#define PYXIS_IO_BASE		(0x8900000000UL)
#define PYXIS_DENSE_MEM		(0x8800000000UL)

/* SABLE is EV4, GAMMA is EV5 */
#define T2_IO_BASE		(0x3a0000000UL)
#define T2_SPARSE_MEM		(0x200000000UL)
#define T2_DENSE_MEM		(0x3c0000000UL)

#define GAMMA_IO_BASE		(0x83a0000000UL)
#define GAMMA_SPARSE_MEM	(0x8200000000UL)
#define GAMMA_DENSE_MEM		(0x83c0000000UL)

/* NOTE: these are hardwired to PCI bus 0 addresses!!! */
#define MCPCIA_IO_BASE		(0xf980000000UL)
#define MCPCIA_SPARSE_MEM	(0xf800000000UL)
#define MCPCIA_DENSE_MEM	(0xf900000000UL)

/* Tsunami and Irongate use the same offsets, at least for hose 0.  */
#define TSUNAMI_IO_BASE		(0x801fc000000UL)
#define TSUNAMI_DENSE_MEM	(0x80000000000UL)

/* Polaris has SPARSE space, but we prefer to use only DENSE
   because of some idiosyncracies in actually using SPARSE.  */
#define POLARIS_IO_BASE		(0xf9fc000000UL)
#define POLARIS_DENSE_MEM	(0xf900000000UL)

typedef enum {
  IOSYS_UNKNOWN, IOSYS_JENSEN, IOSYS_APECS, IOSYS_CIA, IOSYS_PYXIS, IOSYS_T2,
  IOSYS_TSUNAMI, IOSYS_MCPCIA, IOSYS_GAMMA, IOSYS_POLARIS,
  IOSYS_CPUDEP, IOSYS_PCIDEP
} iosys_t;

typedef enum {
  IOSWIZZLE_JENSEN, IOSWIZZLE_SPARSE, IOSWIZZLE_DENSE
} ioswizzle_t;

static struct io_system {
  unsigned long	int bus_memory_base;
  unsigned long	int sparse_bus_mem_base;
  unsigned long	int bus_io_base;
} io_system[] = { /* NOTE! must match iosys_t enumeration */
/* UNKNOWN */	{0, 0, 0},
/* JENSEN */	{0, JENSEN_SPARSE_MEM, JENSEN_IO_BASE},
/* APECS */	{APECS_DENSE_MEM, APECS_SPARSE_MEM, APECS_IO_BASE},
/* CIA */	{CIA_DENSE_MEM, CIA_SPARSE_MEM, CIA_IO_BASE},
/* PYXIS */	{PYXIS_DENSE_MEM, 0, PYXIS_IO_BASE},
/* T2 */	{T2_DENSE_MEM, T2_SPARSE_MEM, T2_IO_BASE},
/* TSUNAMI */	{TSUNAMI_DENSE_MEM, 0, TSUNAMI_IO_BASE},
/* MCPCIA */	{MCPCIA_DENSE_MEM, MCPCIA_SPARSE_MEM, MCPCIA_IO_BASE},
/* GAMMA */	{GAMMA_DENSE_MEM, GAMMA_SPARSE_MEM, GAMMA_IO_BASE},
/* POLARIS */	{POLARIS_DENSE_MEM, 0, POLARIS_IO_BASE},
/* CPUDEP */	{0, 0, 0}, /* for platforms dependent on CPU type */
/* PCIDEP */	{0, 0, 0}, /* for platforms dependent on core logic */
};

static struct platform {
  const char	   *name;
  iosys_t	    io_sys;
} platform[] = {
  {"Alcor",	IOSYS_CIA},
  {"Avanti",	IOSYS_APECS},
  {"Cabriolet",	IOSYS_APECS},
  {"EB164",	IOSYS_PCIDEP},
  {"EB64+",	IOSYS_APECS},
  {"EB66",	IOSYS_APECS},
  {"EB66P",	IOSYS_APECS},
  {"Jensen",	IOSYS_JENSEN},
  {"Miata",	IOSYS_PYXIS},
  {"Mikasa",	IOSYS_CPUDEP},
  {"Nautilus",	IOSYS_TSUNAMI},
  {"Noname",	IOSYS_APECS},
  {"Noritake",	IOSYS_CPUDEP},
  {"Rawhide",	IOSYS_MCPCIA},
  {"Ruffian",	IOSYS_PYXIS},
  {"Sable",	IOSYS_CPUDEP},
  {"Takara",	IOSYS_CIA},
  {"Tsunami",	IOSYS_TSUNAMI},
  {"XL",	IOSYS_APECS},
};

struct ioswtch {
  void		(*sethae)(unsigned long int addr);
  void		(*outb)(unsigned char b, unsigned long int port);
  void		(*outw)(unsigned short b, unsigned long int port);
  void		(*outl)(unsigned int b, unsigned long int port);
  unsigned int	(*inb)(unsigned long int port);
  unsigned int	(*inw)(unsigned long int port);
  unsigned int	(*inl)(unsigned long int port);
};

static struct {
  unsigned long int hae_cache;
  unsigned long int	base;
  struct ioswtch *	swp;
  unsigned long int	bus_memory_base;
  unsigned long int	sparse_bus_memory_base;
  unsigned long int	io_base;
  ioswizzle_t		swiz;
} io;

static inline void
stb_mb(unsigned char val, unsigned long addr)
{
  __asm__("stb %1,%0; mb" : "=m"(*(vucp)addr) : "r"(val));
}

static inline void
stw_mb(unsigned short val, unsigned long addr)
{
  __asm__("stw %1,%0; mb" : "=m"(*(vusp)addr) : "r"(val));
}

static inline void
stl_mb(unsigned int val, unsigned long addr)
{
  __asm__("stl %1,%0; mb" : "=m"(*(vip)addr) : "r"(val));
}

/* No need to examine error -- sethae never fails.  */
static inline void
__sethae(unsigned long value)
{
  INLINE_SYSCALL_CALL (sethae, value);
}

extern long __pciconfig_iobase(enum __pciconfig_iobase_which __which,
			       unsigned long int __bus,
			       unsigned long int __dfn);

static inline unsigned long int
port_to_cpu_addr (unsigned long int port, ioswizzle_t ioswiz, int size)
{
  if (ioswiz == IOSWIZZLE_SPARSE)
    return io.base + (port << 5) + ((size - 1) << 3);
  else if (ioswiz == IOSWIZZLE_DENSE)
    return port + io.base;
  else
    return io.base + (port << 7) + ((size - 1) << 5);
}

static inline __attribute__((always_inline)) void
inline_sethae (unsigned long int addr, ioswizzle_t ioswiz)
{
  if (ioswiz == IOSWIZZLE_SPARSE)
    {
      unsigned long int msb;

      /* no need to set hae if msb is 0: */
      msb = addr & 0xf8000000;
      if (msb && msb != io.hae_cache)
	{
	  io.hae_cache = msb;
	  __sethae (msb);
	}
    }
  else if (ioswiz == IOSWIZZLE_JENSEN)
    {
      /* HAE on the Jensen is bits 31:25 shifted right.  */
      addr >>= 25;
      if (addr != io.hae_cache)
	{
	  io.hae_cache = addr;
	  __sethae (addr);
	}
    }
}

static inline void
inline_outb (unsigned char b, unsigned long int port, ioswizzle_t ioswiz)
{
  unsigned int w;
  unsigned long int addr = port_to_cpu_addr (port, ioswiz, 1);

  asm ("insbl %2,%1,%0" : "=r" (w) : "ri" (port & 0x3), "r" (b));
  stl_mb(w, addr);
}


static inline void
inline_outw (unsigned short int b, unsigned long int port, ioswizzle_t ioswiz)
{
  unsigned long w;
  unsigned long int addr = port_to_cpu_addr (port, ioswiz, 2);

  asm ("inswl %2,%1,%0" : "=r" (w) : "ri" (port & 0x3), "r" (b));
  stl_mb(w, addr);
}


static inline void
inline_outl (unsigned int b, unsigned long int port, ioswizzle_t ioswiz)
{
  unsigned long int addr = port_to_cpu_addr (port, ioswiz, 4);

  stl_mb(b, addr);
}


static inline unsigned int
inline_inb (unsigned long int port, ioswizzle_t ioswiz)
{
  unsigned long int addr = port_to_cpu_addr (port, ioswiz, 1);
  int result;

  result = *(vip) addr;
  result >>= (port & 3) * 8;
  return 0xffUL & result;
}


static inline unsigned int
inline_inw (unsigned long int port, ioswizzle_t ioswiz)
{
  unsigned long int addr = port_to_cpu_addr (port, ioswiz, 2);
  int result;

  result = *(vip) addr;
  result >>= (port & 3) * 8;
  return 0xffffUL & result;
}


static inline unsigned int
inline_inl (unsigned long int port, ioswizzle_t ioswiz)
{
  unsigned long int addr = port_to_cpu_addr (port, ioswiz, 4);

  return *(vuip) addr;
}

/*
 * Now define the inline functions for CPUs supporting byte/word insns,
 * and whose core logic supports I/O space accesses utilizing them.
 *
 * These routines could be used by MIATA, for example, because it has
 * and EV56 plus PYXIS, but it currently uses SPARSE anyway. This is
 * also true of RX164 which used POLARIS, but we will choose to use
 * these routines in that case instead of SPARSE.
 *
 * These routines are necessary for TSUNAMI/TYPHOON based platforms,
 * which will have (at least) EV6.
 */

static inline unsigned long int
dense_port_to_cpu_addr (unsigned long int port)
{
  return port + io.base;
}

static inline void
inline_bwx_outb (unsigned char b, unsigned long int port)
{
  unsigned long int addr = dense_port_to_cpu_addr (port);
  stb_mb (b, addr);
}

static inline void
inline_bwx_outw (unsigned short int b, unsigned long int port)
{
  unsigned long int addr = dense_port_to_cpu_addr (port);
  stw_mb (b, addr);
}

static inline void
inline_bwx_outl (unsigned int b, unsigned long int port)
{
  unsigned long int addr = dense_port_to_cpu_addr (port);
  stl_mb (b, addr);
}

static inline unsigned int
inline_bwx_inb (unsigned long int port)
{
  unsigned long int addr = dense_port_to_cpu_addr (port);
  unsigned char r;

  __asm__ ("ldbu %0,%1" : "=r"(r) : "m"(*(vucp)addr));
  return r;
}

static inline unsigned int
inline_bwx_inw (unsigned long int port)
{
  unsigned long int addr = dense_port_to_cpu_addr (port);
  unsigned short r;

  __asm__ ("ldwu %0,%1" : "=r"(r) : "m"(*(vusp)addr));
  return r;
}

static inline unsigned int
inline_bwx_inl (unsigned long int port)
{
  unsigned long int addr = dense_port_to_cpu_addr (port);

  return *(vuip) addr;
}

/* macros to define routines with appropriate names and functions */

/* these do either SPARSE or JENSEN swizzle */

#define DCL_SETHAE(name, ioswiz)                        \
static void                                             \
name##_sethae (unsigned long int addr)                  \
{                                                       \
  inline_sethae (addr, IOSWIZZLE_##ioswiz);             \
}

#define DCL_OUT(name, func, type, ioswiz)		\
static void						\
name##_##func (unsigned type b, unsigned long int addr)	\
{							\
  inline_##func (b, addr, IOSWIZZLE_##ioswiz);		\
}

#define DCL_IN(name, func, ioswiz)			\
static unsigned int					\
name##_##func (unsigned long int addr)			\
{							\
  return inline_##func (addr, IOSWIZZLE_##ioswiz);	\
}

/* these do DENSE, so no swizzle is needed */

#define DCL_OUT_BWX(name, func, type)			\
static void						\
name##_##func (unsigned type b, unsigned long int addr)	\
{							\
  inline_bwx_##func (b, addr);				\
}

#define DCL_IN_BWX(name, func)				\
static unsigned int					\
name##_##func (unsigned long int addr)			\
{							\
  return inline_bwx_##func (addr);			\
}

/* now declare/define the necessary routines */

DCL_SETHAE(jensen, JENSEN)
DCL_OUT(jensen, outb, char,  JENSEN)
DCL_OUT(jensen, outw, short int, JENSEN)
DCL_OUT(jensen, outl, int,   JENSEN)
DCL_IN(jensen, inb, JENSEN)
DCL_IN(jensen, inw, JENSEN)
DCL_IN(jensen, inl, JENSEN)

DCL_SETHAE(sparse, SPARSE)
DCL_OUT(sparse, outb, char,  SPARSE)
DCL_OUT(sparse, outw, short int, SPARSE)
DCL_OUT(sparse, outl, int,   SPARSE)
DCL_IN(sparse, inb, SPARSE)
DCL_IN(sparse, inw, SPARSE)
DCL_IN(sparse, inl, SPARSE)

DCL_SETHAE(dense, DENSE)
DCL_OUT_BWX(dense, outb, char)
DCL_OUT_BWX(dense, outw, short int)
DCL_OUT_BWX(dense, outl, int)
DCL_IN_BWX(dense, inb)
DCL_IN_BWX(dense, inw)
DCL_IN_BWX(dense, inl)

/* define the "swizzle" switch */
static struct ioswtch ioswtch[] = {
  {
    jensen_sethae,
    jensen_outb, jensen_outw, jensen_outl,
    jensen_inb, jensen_inw, jensen_inl
  },
  {
    sparse_sethae,
    sparse_outb, sparse_outw, sparse_outl,
    sparse_inb, sparse_inw, sparse_inl
  },
  {
    dense_sethae,
    dense_outb, dense_outw, dense_outl,
    dense_inb, dense_inw, dense_inl
  }
};

#undef DEBUG_IOPERM

/* Routine to process the /proc/cpuinfo information into the fields
   that are required for correctly determining the platform parameters.  */

struct cpuinfo_data
{
  char systype[256];		/* system type field */
  char sysvari[256];		/* system variation field */
  char cpumodel[256];		/* cpu model field */
};

static inline int
process_cpuinfo(struct cpuinfo_data *data)
{
  int got_type, got_vari, got_model;
  char dummy[256];
  FILE * fp;
  int n;

  data->systype[0] = 0;
  data->sysvari[0] = 0;
  data->cpumodel[0] = 0;

  /* If there's an /etc/alpha_systype link, we're intending to override
     whatever's in /proc/cpuinfo.  */
  n = __readlink (PATH_ALPHA_SYSTYPE, data->systype, 256 - 1);
  if (n > 0)
    {
      data->systype[n] = '\0';
      return 1;
    }

  fp = fopen (PATH_CPUINFO, "rce");
  if (!fp)
    return 0;

  got_type = got_vari = got_model = 0;

  while (1)
    {
      if (fgets_unlocked (dummy, 256, fp) == NULL)
	break;
      if (!got_type
	  && sscanf (dummy, "system type : %256[^\n]\n", data->systype) == 1)
	got_type = 1;
      if (!got_vari
	  && (sscanf (dummy, "system variation : %256[^\n]\n", data->sysvari)
	      == 1))
	got_vari = 1;
      if (!got_model
	  && sscanf (dummy, "cpu model : %256[^\n]\n", data->cpumodel) == 1)
	got_model = 1;
    }

  fclose (fp);

#ifdef DEBUG_IOPERM
  fprintf(stderr, "system type: `%s'\n", data->systype);
  fprintf(stderr, "system vari: `%s'\n", data->sysvari);
  fprintf(stderr, "cpu model: `%s'\n", data->cpumodel);
#endif

  return got_type + got_vari + got_model;
}


/*
 * Initialize I/O system.
 */
static int
init_iosys (void)
{
  long addr;
  int i, olderrno = errno;
  struct cpuinfo_data data;

  /* First try the pciconfig_iobase syscall added to 2.2.15 and 2.3.99.  */

  addr = __pciconfig_iobase (IOBASE_DENSE_MEM, 0, 0);
  if (addr != -1)
    {
      ioswizzle_t io_swiz;

      if (addr == 0)
        {
	  /* Only Jensen doesn't have dense mem space.  */
	  io.sparse_bus_memory_base
	    = io_system[IOSYS_JENSEN].sparse_bus_mem_base;
	  io.io_base = io_system[IOSYS_JENSEN].bus_io_base;
	  io_swiz = IOSWIZZLE_JENSEN;
	}
      else
	{
	  io.bus_memory_base = addr;

	  addr = __pciconfig_iobase (IOBASE_DENSE_IO, 0, 0);
	  if (addr != 0)
	    {
	      /* The X server uses _bus_base_sparse == 0 to know that
		 BWX access are supported to dense mem space.  This is
		 true of every system that supports dense io space, so
	         never fill in io.sparse_bus_memory_base in this case.  */
	      io_swiz = IOSWIZZLE_DENSE;
              io.io_base = addr;
	    }
	  else
	    {
	      io.sparse_bus_memory_base
		= __pciconfig_iobase (IOBASE_SPARSE_MEM, 0, 0);
	      io.io_base = __pciconfig_iobase (IOBASE_SPARSE_IO, 0, 0);
	      io_swiz = IOSWIZZLE_SPARSE;
	    }
	}

      io.swiz = io_swiz;
      io.swp = &ioswtch[io_swiz];

      return 0;
    }

  /* Second, collect the contents of /etc/alpha_systype or /proc/cpuinfo.  */

  if (process_cpuinfo(&data) == 0)
    {
      /* This can happen if the format of /proc/cpuinfo changes.  */
      fprintf (stderr,
	       "ioperm.init_iosys: Unable to determine system type.\n"
	       "\t(May need " PATH_ALPHA_SYSTYPE " symlink?)\n");
      __set_errno (ENODEV);
      return -1;
    }

  /* Translate systype name into i/o system.  */
  for (i = 0; i < sizeof (platform) / sizeof (platform[0]); ++i)
    {
      if (strcmp (platform[i].name, data.systype) == 0)
	{
	  iosys_t io_sys = platform[i].io_sys;

	  /* Some platforms can have either EV4 or EV5 CPUs.  */
	  if (io_sys == IOSYS_CPUDEP)
	    {
	      /* SABLE or MIKASA or NORITAKE so far.  */
	      if (strcmp (platform[i].name, "Sable") == 0)
		{
		  if (strncmp (data.cpumodel, "EV4", 3) == 0)
		    io_sys = IOSYS_T2;
		  else if (strncmp (data.cpumodel, "EV5", 3) == 0)
		    io_sys = IOSYS_GAMMA;
		}
	      else
		{
		  /* This covers MIKASA/NORITAKE.  */
		  if (strncmp (data.cpumodel, "EV4", 3) == 0)
		    io_sys = IOSYS_APECS;
		  else if (strncmp (data.cpumodel, "EV5", 3) == 0)
		    io_sys = IOSYS_CIA;
		}
	      if (io_sys == IOSYS_CPUDEP)
		{
		  /* This can happen if the format of /proc/cpuinfo changes.*/
		  fprintf (stderr, "ioperm.init_iosys: Unable to determine"
			   " CPU model.\n");
		  __set_errno (ENODEV);
		  return -1;
		}
	    }
	  /* Some platforms can have different core logic chipsets */
	  if (io_sys == IOSYS_PCIDEP)
	    {
	      /* EB164 so far */
	      if (strcmp (data.systype, "EB164") == 0)
		{
		  if (strncmp (data.sysvari, "RX164", 5) == 0)
		    io_sys = IOSYS_POLARIS;
		  else if (strncmp (data.sysvari, "LX164", 5) == 0
			   || strncmp (data.sysvari, "SX164", 5) == 0)
		    io_sys = IOSYS_PYXIS;
		  else
		    io_sys = IOSYS_CIA;
		}
	      if (io_sys == IOSYS_PCIDEP)
		{
		  /* This can happen if the format of /proc/cpuinfo changes.*/
		  fprintf (stderr, "ioperm.init_iosys: Unable to determine"
			   " core logic chipset.\n");
		  __set_errno (ENODEV);
		  return -1;
		}
	    }
	  io.bus_memory_base = io_system[io_sys].bus_memory_base;
	  io.sparse_bus_memory_base = io_system[io_sys].sparse_bus_mem_base;
	  io.io_base = io_system[io_sys].bus_io_base;

	  if (io_sys == IOSYS_JENSEN)
	    io.swiz = IOSWIZZLE_JENSEN;
	  else if (io_sys == IOSYS_TSUNAMI
		   || io_sys == IOSYS_POLARIS
		   || io_sys == IOSYS_PYXIS)
	    io.swiz = IOSWIZZLE_DENSE;
	  else
	    io.swiz = IOSWIZZLE_SPARSE;
	  io.swp = &ioswtch[io.swiz];

	  __set_errno (olderrno);
	  return 0;
	}
    }

  __set_errno (ENODEV);
  fprintf(stderr, "ioperm.init_iosys: Platform not recognized.\n"
	  "\t(May need " PATH_ALPHA_SYSTYPE " symlink?)\n");
  return -1;
}


int
_ioperm (unsigned long int from, unsigned long int num, int turn_on)
{
  unsigned long int addr, len, pagesize = __getpagesize();
  int prot;

  if (!io.swp && init_iosys() < 0)
    {
#ifdef DEBUG_IOPERM
      fprintf(stderr, "ioperm: init_iosys() failed (%m)\n");
#endif
      return -1;
    }

  /* This test isn't as silly as it may look like; consider overflows! */
  if (from >= MAX_PORT || from + num > MAX_PORT)
    {
      __set_errno (EINVAL);
#ifdef DEBUG_IOPERM
      fprintf(stderr, "ioperm: from/num out of range\n");
#endif
      return -1;
    }

#ifdef DEBUG_IOPERM
  fprintf(stderr, "ioperm: turn_on %d io.base %ld\n", turn_on, io.base);
#endif

  if (turn_on)
    {
      if (!io.base)
	{
	  int fd;

	  io.hae_cache = 0;
	  if (io.swiz != IOSWIZZLE_DENSE)
	    {
	      /* Synchronize with hw.  */
	      __sethae (0);
	    }

	  fd = __open ("/dev/mem", O_RDWR);
	  if (fd < 0)
	    {
#ifdef DEBUG_IOPERM
	      fprintf(stderr, "ioperm: /dev/mem open failed (%m)\n");
#endif
	      return -1;
	    }

	  addr = port_to_cpu_addr (0, io.swiz, 1);
	  len = port_to_cpu_addr (MAX_PORT, io.swiz, 1) - addr;
	  io.base =
	    (unsigned long int) __mmap (0, len, PROT_NONE, MAP_SHARED,
					fd, io.io_base);
	  __close (fd);
#ifdef DEBUG_IOPERM
	  fprintf(stderr, "ioperm: mmap of len 0x%lx  returned 0x%lx\n",
		  len, io.base);
#endif
	  if ((long) io.base == -1)
	    return -1;
	}
      prot = PROT_READ | PROT_WRITE;
    }
  else
    {
      if (!io.base)
	return 0;	/* never was turned on... */

      /* turnoff access to relevant pages: */
      prot = PROT_NONE;
    }
  addr = port_to_cpu_addr (from, io.swiz, 1);
  addr &= ~(pagesize - 1);
  len = port_to_cpu_addr (from + num, io.swiz, 1) - addr;
  return __mprotect ((void *) addr, len, prot);
}


int
_iopl (int level)
{
  switch (level)
    {
    case 0:
      return 0;

    case 1: case 2: case 3:
      return _ioperm (0, MAX_PORT, 1);

    default:
      __set_errno (EINVAL);
      return -1;
    }
}


void
_sethae (unsigned long int addr)
{
  if (!io.swp && init_iosys () < 0)
    return;

  io.swp->sethae (addr);
}


void
_outb (unsigned char b, unsigned long int port)
{
  if (port >= MAX_PORT)
    return;

  io.swp->outb (b, port);
}


void
_outw (unsigned short b, unsigned long int port)
{
  if (port >= MAX_PORT)
    return;

  io.swp->outw (b, port);
}


void
_outl (unsigned int b, unsigned long int port)
{
  if (port >= MAX_PORT)
    return;

  io.swp->outl (b, port);
}


unsigned int
_inb (unsigned long int port)
{
  return io.swp->inb (port);
}


unsigned int
_inw (unsigned long int port)
{
  return io.swp->inw (port);
}


unsigned int
_inl (unsigned long int port)
{
  return io.swp->inl (port);
}


unsigned long int
_bus_base(void)
{
  if (!io.swp && init_iosys () < 0)
    return -1;
  return io.bus_memory_base;
}

unsigned long int
_bus_base_sparse(void)
{
  if (!io.swp && init_iosys () < 0)
    return -1;
  return io.sparse_bus_memory_base;
}

int
_hae_shift(void)
{
  if (!io.swp && init_iosys () < 0)
    return -1;
  if (io.swiz == IOSWIZZLE_JENSEN)
    return 7;
  if (io.swiz == IOSWIZZLE_SPARSE)
    return 5;
  return 0;
}

weak_alias (_sethae, sethae);
weak_alias (_ioperm, ioperm);
weak_alias (_iopl, iopl);
weak_alias (_inb, inb);
weak_alias (_inw, inw);
weak_alias (_inl, inl);
weak_alias (_outb, outb);
weak_alias (_outw, outw);
weak_alias (_outl, outl);
weak_alias (_bus_base, bus_base);
weak_alias (_bus_base_sparse, bus_base_sparse);
weak_alias (_hae_shift, hae_shift);

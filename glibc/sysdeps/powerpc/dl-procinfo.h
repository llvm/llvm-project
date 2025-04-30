/* Processor capability information handling macros.  PowerPC version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#ifndef _DL_PROCINFO_H
#define _DL_PROCINFO_H 1

#include <ldsodefs.h>
#include <sysdep.h>	/* This defines the PPC_FEATURE[2]_* macros.  */

/* The total number of available bits (including those prior to
   _DL_HWCAP_FIRST).  Some of these bits might not be used.  */
#define _DL_HWCAP_COUNT		64

/* Features started at bit 31 and decremented as new features were added.  */
#define _DL_HWCAP_LAST		31

/* AT_HWCAP2 features started at bit 31 and decremented as new features were
   added.  HWCAP2 feature bits start at bit 0.  */
#define _DL_HWCAP2_LAST		31

/* These bits influence library search.  */
#define HWCAP_IMPORTANT		(PPC_FEATURE_HAS_ALTIVEC \
				+ PPC_FEATURE_HAS_DFP)

#define _DL_PLATFORMS_COUNT	16

#define _DL_FIRST_PLATFORM	32
/* Mask to filter out platforms.  */
#define _DL_HWCAP_PLATFORM	(((1ULL << _DL_PLATFORMS_COUNT) - 1) \
				<< _DL_FIRST_PLATFORM)

/* Platform bits (relative to _DL_FIRST_PLATFORM).  */
#define PPC_PLATFORM_POWER4		0
#define PPC_PLATFORM_PPC970		1
#define PPC_PLATFORM_POWER5		2
#define PPC_PLATFORM_POWER5_PLUS	3
#define PPC_PLATFORM_POWER6		4
#define PPC_PLATFORM_CELL_BE		5
#define PPC_PLATFORM_POWER6X		6
#define PPC_PLATFORM_POWER7		7
#define PPC_PLATFORM_PPCA2		8
#define PPC_PLATFORM_PPC405		9
#define PPC_PLATFORM_PPC440		10
#define PPC_PLATFORM_PPC464		11
#define PPC_PLATFORM_PPC476		12
#define PPC_PLATFORM_POWER8		13
#define PPC_PLATFORM_POWER9		14
#define PPC_PLATFORM_POWER10		15

static inline const char *
__attribute__ ((unused))
_dl_hwcap_string (int idx)
{
  return GLRO(dl_powerpc_cap_flags)[idx];
}

static inline int
__attribute__ ((unused))
_dl_string_hwcap (const char *str)
{
  for (int i = 0; i < _DL_HWCAP_COUNT; ++i)
    if (strcmp (str, _dl_hwcap_string (i)) == 0)
      return i;
  return -1;
}

static inline int
__attribute__ ((unused, always_inline))
_dl_string_platform (const char *str)
{
  if (str == NULL)
    return -1;

  if (strncmp (str, "power", 5) == 0)
    {
      int ret;
      str += 5;
      switch (*str)
	{
	case '1':
	  if (str[1] == '0')
	    {
	      ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER10;
	      str++;
	    }
	  else
	    return -1;
	  break;
	case '4':
	  ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER4;
	  break;
	case '5':
	  ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER5;
	  if (str[1] == '+')
	    {
	      ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER5_PLUS;
	      ++str;
	    }
	  break;
	case '6':
	  ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER6;
	  if (str[1] == 'x')
	    {
	      ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER6X;
	      ++str;
	    }
	  break;
	case '7':
	  ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER7;
	  break;
	case '8':
	  ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER8;
	  break;
	case '9':
	  ret = _DL_FIRST_PLATFORM + PPC_PLATFORM_POWER9;
	  break;
	default:
	  return -1;
	}
      if (str[1] == '\0')
       return ret;
    }
  else if (strncmp (str, "ppc", 3) == 0)
    {
      if (strcmp (str + 3, "970") == 0)
	return _DL_FIRST_PLATFORM + PPC_PLATFORM_PPC970;
      else if (strcmp (str + 3, "-cell-be") == 0)
	return _DL_FIRST_PLATFORM + PPC_PLATFORM_CELL_BE;
      else if (strcmp (str + 3, "a2") == 0)
	return _DL_FIRST_PLATFORM + PPC_PLATFORM_PPCA2;
      else if (strcmp (str + 3, "405") == 0)
	return _DL_FIRST_PLATFORM + PPC_PLATFORM_PPC405;
      else if (strcmp (str + 3, "440") == 0)
	return _DL_FIRST_PLATFORM + PPC_PLATFORM_PPC440;
      else if (strcmp (str + 3, "464") == 0)
	return _DL_FIRST_PLATFORM + PPC_PLATFORM_PPC464;
      else if (strcmp (str + 3, "476") == 0)
	return _DL_FIRST_PLATFORM + PPC_PLATFORM_PPC476;
    }

  return -1;
}

#if IS_IN (rtld)
static inline void
cache_geometry (const char * name, unsigned long int geometry)
{
  unsigned long int assocty, line;

  _dl_printf ("%s", name);

  line = geometry & 0xffff;
  assocty = (geometry >> 16) & 0xffff;

  if (line == 0)
    _dl_printf ("Unknown line size, ");
  else
    _dl_printf ("%luB line size, ", line);

  switch (assocty)
    {
    case 0:
      _dl_printf ("Unknown associativity");
      break;
    case 1:
      _dl_printf ("Directly mapped");
      break;
    case 0xffff:
      _dl_printf ("Fully associative");
      break;
    default:
      _dl_printf ("%lu-way set associative", assocty);
    }
}

static inline int
__attribute__ ((unused))
_dl_procinfo (unsigned int type, unsigned long int word)
{
  switch(type)
    {
    case AT_HWCAP:
      _dl_printf ("AT_HWCAP:            ");

      for (int i = 0; i <= _DL_HWCAP_LAST; ++i)
       if (word & (1 << i))
         _dl_printf (" %s", _dl_hwcap_string (i));
      break;
    case AT_HWCAP2:
      {
       unsigned int offset = _DL_HWCAP_LAST + 1;

       _dl_printf ("AT_HWCAP2:           ");

        /* We have to go through them all because the kernel added the
          AT_HWCAP2 features starting with the high bits.  */
       for (int i = 0; i <= _DL_HWCAP2_LAST; ++i)
         if (word & (1 << i))
           _dl_printf (" %s", _dl_hwcap_string (offset + i));
       break;
      }
    case AT_L1I_CACHEGEOMETRY:
      {
	cache_geometry ("AT_L1I_CACHEGEOMETRY: ", word);
	break;
      }
    case AT_L1D_CACHEGEOMETRY:
      {
	cache_geometry ("AT_L1D_CACHEGEOMETRY: ", word);
	break;
      }
    case AT_L2_CACHEGEOMETRY:
      {
	cache_geometry ("AT_L2_CACHEGEOMETRY:  ", word);
	break;
      }
    case AT_L3_CACHEGEOMETRY:
      {
	cache_geometry ("AT_L3_CACHEGEOMETRY:  ", word);
	break;
      }
    default:
      /* Fallback to generic output mechanism.  */
      return -1;
    }
   _dl_printf ("\n");
  return 0;
}
#endif

#endif /* dl-procinfo.h */

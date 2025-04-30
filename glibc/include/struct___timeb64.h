#ifndef _STRUCT_TIMEB64_H
#define _STRUCT_TIMEB64_H

#if __TIMESIZE == 64
# define __timeb64 timeb
#else
struct __timeb64
{
  __time64_t time;
  unsigned short int millitm;
  short int timezone;
  short int dstflag;
};
#endif

#endif /* _STRUCT_TIMEB64_H  */

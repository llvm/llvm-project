#ifndef _SYS_STAT_H
#define _SYS_STAT_H

#include "textual_time.h"

struct stat {
  struct timespec st_atim;
  struct timespec st_mtim;
};

#endif

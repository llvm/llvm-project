
#ifndef TARGETED_TOP_H
#define TARGETED_TOP_H

#include "targeted-nested1.h"

enum {
  VALUE = 3
};

extern int TopVar;

typedef struct {
  int x;
  int y;
#include "targeted-fields.h"
} Vector;

static inline int vector_get_x(Vector v) {
  int x = v.x;
  return x;
}

#endif

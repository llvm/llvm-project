#ifndef H_H
#define H_H
#include "c.h"
#include "d.h" // expected-error {{module XH does not directly depend on a module exporting 'd.h', which is part of indirectly-used module XD}}
#include "h1.h"
const int h1 = aux_h*c*7*d;
#endif

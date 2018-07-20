#ifndef H_H
#define H_H
#include "c.h"
#include "d.h" // expected-error {{module XH does not depend on a module exporting}}
#include "h1.h"
const int h1 = aux_h*c*7*d;
#endif

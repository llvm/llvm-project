#include "A.h"

int branch;

int useAlias(const A &alias) {
    int value;
    if (branch > 0)
        value = alias.getValue();
    else
        value = 0;
    return value;
}
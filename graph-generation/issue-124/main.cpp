#include "stdio.h"
int g_table;

int *func2(int id)
{
    if (id >= 2) {
        return NULL;
    }

    return (int *)&g_table;
}

int func1(int id)
{
    int *ptr = func2(id);

    if(1) {}

    return (*ptr);
}

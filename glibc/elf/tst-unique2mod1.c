#include <config.h>

asm (".data;"
     ".globl var\n"
     ".type var, %gnu_unique_object\n"
     ".size var, 4\n"
     "var:.zero 4\n"
     ".previous");

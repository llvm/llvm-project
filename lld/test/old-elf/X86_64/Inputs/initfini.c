#include <stdio.h>

void  __attribute__ ((constructor)) constructor() {
 printf("%s\n", __FUNCTION__);
}

void __attribute__ ((destructor)) destructor() {
 printf("%s\n", __FUNCTION__);
}

int main() {
  return 0;
}


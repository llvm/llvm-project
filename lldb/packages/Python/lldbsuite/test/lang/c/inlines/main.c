#include <stdio.h>

inline void test1(int) __attribute__ ((always_inline));
inline void test2(int) __attribute__ ((always_inline));

void test2(int b) {
    printf("test2(%d)\n", b); //% self.expect("expression b", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["42"])
}

void test1(int a) {
    printf("test1(%d)\n",  a);
    test2(a+1);//% self.dbg.HandleCommand("step")
               //% self.expect("expression b", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["24"])
}

int main() {
    test2(42);
    test1(23);
}

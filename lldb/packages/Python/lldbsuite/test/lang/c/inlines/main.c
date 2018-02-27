#include <stdio.h>

inline void test1(int) __attribute__ ((always_inline));
inline void test2(int) __attribute__ ((always_inline));

void test2(int b) {
    printf("test2(%d)\n", b); //% self.expect("expression b", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["42"])
    {
      int c = b * 2;
      printf("c=%d\n", c); //% self.expect("expression b", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["42"])
                           //% self.expect("expression c", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["84"])
    }
}

void test1(int a) {
    printf("test1(%d)\n",  a);
    test2(a+1);//% self.runCmd("step")
               //% self.expect("expression b", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["24"])
}

int main() {
    test2(42);
    test1(23);
    return 0;
}

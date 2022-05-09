// RUN: %clang_cc1 -emit-llvm %s -o %t

/* Sparc is not C99-compliant */
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)

int main(void) { return 0; }

#else /* sparc */

#define ESCAPE 2
#define IMAGE_WIDTH 150
#define IMAGE_HEIGHT 50
#if 1
#define IMAGE_SIZE 60
#else
#define IMAGE_SIZE 5000
#endif
#define START_X -2.1
#define END_X 1.0
#define START_Y -1.25
#define MAX_ITER 100

#define step_X ((END_X - START_X)/IMAGE_WIDTH)
#define step_Y ((-START_Y - START_Y)/IMAGE_HEIGHT)

#define I 1.0iF

int putchar(char c);
double hypot(double, double);

volatile double __complex__ accum;

void mandel(void) {
  int x, y, n;
  for (y = 0; y < IMAGE_HEIGHT; ++y) {
    for (x = 0; x < IMAGE_WIDTH; ++x) {
      double __complex__ c = (START_X+x*step_X) + (START_Y+y*step_Y) * I;
      double __complex__ z = 0.0;

      for (n = 0; n < MAX_ITER; ++n) {
        z = z * z + c;
        if (hypot(__real__ z, __imag__ z) >= ESCAPE)
          break;
      }

      if (n == MAX_ITER)
        putchar(' ');
      else if (n > 6)
        putchar('.');
      else if (n > 3)
        putchar('+');
      else if (n > 2)
        putchar('x');
      else
        putchar('*');
    }
    putchar('\n');
  }
}

int main(void) {
  mandel();
  return 0;
}

#endif /* sparc */

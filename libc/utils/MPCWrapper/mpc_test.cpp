#include "mpc.h"

int main() {
  // just a pair of two arbitary precision floating point numbers
  mpc_t x, z;
  // different rounding modes for real and complex part
  mpc_rnd_t y;

  // MPC function begins with mpc_ and store the result in the first argument.
  // Rounding modes in MPC are of the form MPC_RNDxy where x and y are one of :
  //     1. N(to nearest)
  //     2. Z(to zero)
  //     3. U(towards +infinity)
  //     4. D(towards -infinity)
  //     5. A(away from zero)
  // where the first letter is for the real part and the second letter is for
  // the imaginary part. MPC functions have a return value which can be used
  // with MPC_INEX_RE and MPC_INEX_IM to tell whether the rounded value is less,
  // equal or greater than the exact value. Results on machines with different
  // word sizes should not wary.

  // for multiplication.
  mpc_mul(x, x, x, MPC_RNDNN);

  // for initialization
  mpc_init2(x, 256);      // Real and Imaginary have same precision
  mpc_init3(x, 256, 256); // Real and Imaginary can have different precision
  mpc_clear(x);           // free the space occupied by mpc_t x
  mpc_set_prec(x, 64); // reset the precision of x to 64 bits and set x to (NaN,
                       // NaN) (Previous value of x is lost)
  // mpc_get_prec returns the precision if precision of real and imaginary part
  // of x are same, else it returns 0. mpc_get_prec2 returns the precision of
  // the real and imaginary part of x.
  mpc_set(x, z, MPC_RNDNN); // set x to z with rounding mode MPC_RNDNN
}

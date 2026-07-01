// Stage 2c: load-order independence. Same as use_err.cpp but with the include
// order flipped so B is seen first. Whichever module loads first becomes the
// canonical copy; the other redirects into it. Result should be identical:
// dedup active + the diagnostic still resolves to shared.h:<line>.
#include "b.h"
#include "a.h"

int main() {
  return a_entry(1) + shared_fn_1("oops");
}

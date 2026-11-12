// RUN: %clang_cc1 %s -emit-llvm-only
// CHECK that we don't crash.

int test(int val){
	int x = 12;
	// Make sure we don't crash when constant folding the case 4
	// statement due to the case 5 statement contained in the do loop
	switch (val) {
		case 4: do { 
                     switch (6)  {
                       case 6: {
                         case 5: x++;
                       };
                     };
                } while (x < 100);
	}
	return x;
}

int main(void) {
  return test(4);
}

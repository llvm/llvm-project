// RUN: %check_clang_tidy %s llvm-prefer-register-over-unsigned %t

namespace llvm { };

// This class shouldn't trigger it despite the similarity as it's not inside the llvm namespace
class Register {
public:
  operator unsigned();
};

Register getReg();

void do_nothing_1() {
  unsigned Reg1 = getReg();
  // CHECK-FIXES: do_nothing_1()
  // CHECK-FIXES-NEXT: unsigned Reg1 = getReg();
}

void do_nothing_2() {
  using namespace llvm;
  unsigned Reg2 = getReg();
  // CHECK-FIXES: do_nothing_2()
  // CHECK-FIXES-NEXT: using namespace llvm;
  // CHECK-FIXES-NEXT: unsigned Reg2 = getReg();
}

namespace llvm {
void do_nothing_3() {
  unsigned Reg3 = getReg();
  // CHECK-FIXES: do_nothing_3()
  // CHECK-FIXES-NEXT: unsigned Reg3 = getReg();
}
} // end namespace llvm

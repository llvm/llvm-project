// RUN: %check_clang_tidy %s llvm-prefer-register-over-unsigned %t

namespace llvm {
class Register {
public:
  operator unsigned();
};
} // end namespace llvm

llvm::Register getReg();

using namespace llvm;

void apply_1() {
  unsigned Reg = getReg();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'Reg' declared as 'unsigned int'; use 'Register' instead [llvm-prefer-register-over-unsigned]
  // CHECK-FIXES: apply_1()
  // CHECK-FIXES-NEXT: Register Reg = getReg();
}

void done_1() {
  llvm::Register Reg = getReg();
  // CHECK-FIXES: done_1()
  // CHECK-FIXES-NEXT: llvm::Register Reg = getReg();
}

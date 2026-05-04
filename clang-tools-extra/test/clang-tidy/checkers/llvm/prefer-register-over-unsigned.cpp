// RUN: %check_clang_tidy %s llvm-prefer-register-over-unsigned %t

namespace llvm {
class Register {
public:
  operator unsigned();

  unsigned Reg;
};

// This class shouldn't trigger it despite the similarity.
class RegisterLike {
public:
  operator unsigned();

  unsigned Reg;
};
} // end namespace llvm

llvm::Register getReg();
llvm::RegisterLike getRegLike();

void apply_1() {
  unsigned Reg1 = getReg();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'Reg1' declared as 'unsigned int'; use 'llvm::Register' instead [llvm-prefer-register-over-unsigned]
  // CHECK-FIXES:      void apply_1() {
  // CHECK-FIXES-NEXT:   llvm::Register Reg1 = getReg();
}

void apply_2() {
  using namespace llvm;
  unsigned Reg2 = getReg();
  // FIXME: Function-scoped UsingDirectiveDecl's don't appear to be in the
  //        DeclContext for the function so we can't elide the llvm:: in this
  //        case. Fortunately, it doesn't actually occur in the LLVM code base.
  // CHECK-MESSAGES: :[[@LINE-4]]:12: warning: variable 'Reg2' declared as 'unsigned int'; use 'llvm::Register' instead [llvm-prefer-register-over-unsigned]
  // CHECK-FIXES:      void apply_2() {
  // CHECK-FIXES-NEXT:   using namespace llvm;
  // CHECK-FIXES-NEXT:   llvm::Register Reg2 = getReg();
}

namespace llvm {
void apply_3() {
  unsigned Reg3 = getReg();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'Reg3' declared as 'unsigned int'; use 'Register' instead [llvm-prefer-register-over-unsigned]
  // CHECK-FIXES:      void apply_3() {
  // CHECK-FIXES-NEXT:   Register Reg3 = getReg();
}
} // end namespace llvm

void done_1() {
  llvm::Register Reg1 = getReg();
  // CHECK-FIXES:      void done_1() {
  // CHECK-FIXES-NEXT:   llvm::Register Reg1 = getReg();
}

void done_2() {
  using namespace llvm;
  Register Reg2 = getReg();
  // CHECK-FIXES:      void done_2() {
  // CHECK-FIXES-NEXT:   using namespace llvm;
  // CHECK-FIXES-NEXT:   Register Reg2 = getReg();
}

namespace llvm {
void done_3() {
  Register Reg3 = getReg();
  // CHECK-FIXES:      void done_3() {
  // CHECK-FIXES-NEXT:   Register Reg3 = getReg();
}
} // end namespace llvm

void do_nothing_1() {
  unsigned Reg1 = getRegLike();
  // CHECK-FIXES:      void do_nothing_1() {
  // CHECK-FIXES-NEXT:   unsigned Reg1 = getRegLike();
}

void do_nothing_2() {
  using namespace llvm;
  unsigned Reg2 = getRegLike();
  // CHECK-FIXES:      void do_nothing_2() {
  // CHECK-FIXES-NEXT:   using namespace llvm;
  // CHECK-FIXES-NEXT:   unsigned Reg2 = getRegLike();
}

namespace llvm {
void do_nothing_3() {
  unsigned Reg3 = getRegLike();
  // CHECK-FIXES:      void do_nothing_3() {
  // CHECK-FIXES-NEXT:   unsigned Reg3 = getRegLike();
}
} // end namespace llvm

void fn1(llvm::Register R);
void do_nothing_4() {
  fn1(getReg());
  // CHECK-FIXES:      void do_nothing_4() {
  // CHECK-FIXES-NEXT:   fn1(getReg());
}

void fn2(unsigned R);
void do_nothing_5() {
  fn2(getReg());
  // CHECK-FIXES:      void do_nothing_5() {
  // CHECK-FIXES-NEXT:   fn2(getReg());
}

void do_nothing_6() {
  using namespace llvm;
  Register Reg6{getReg()};
  // CHECK-FIXES:      void do_nothing_6() {
  // CHECK-FIXES-NEXT:   using namespace llvm;
  // CHECK-FIXES-NEXT:   Register Reg6{getReg()};
}

void do_nothing_7() {
  using namespace llvm;
  Register Reg7;
  Reg7.Reg = getReg();
  // CHECK-FIXES:      void do_nothing_7() {
  // CHECK-FIXES-NEXT:   using namespace llvm;
  // CHECK-FIXES-NEXT:   Register Reg7;
  // CHECK-FIXES-NEXT:   Reg7.Reg = getReg();
}

void do_nothing_8() {
  using namespace llvm;
  RegisterLike Reg8{getReg()};
  // CHECK-FIXES:      void do_nothing_8() {
  // CHECK-FIXES-NEXT:   using namespace llvm;
  // CHECK-FIXES-NEXT:   RegisterLike Reg8{getReg()};
}

void do_nothing_9() {
  using namespace llvm;
  RegisterLike Reg9;
  Reg9.Reg = getReg();
  // CHECK-FIXES:      void do_nothing_9() {
  // CHECK-FIXES-NEXT:   using namespace llvm;
  // CHECK-FIXES-NEXT:   RegisterLike Reg9;
  // CHECK-FIXES-NEXT:   Reg9.Reg = getReg();
}

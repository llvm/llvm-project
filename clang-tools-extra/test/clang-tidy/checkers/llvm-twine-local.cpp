// RUN: %check_clang_tidy %s llvm-twine-local %t

namespace llvm {
class Twine {
public:
  Twine(const char *);
  Twine(int);
  Twine();
  Twine &operator+(const Twine &);
};
}

using namespace llvm;

void foo(const Twine &x);
void bar(Twine x);

static Twine Moo = Twine("bark") + "bah";
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: static std::string Moo = (Twine("bark") + "bah").str();

int main() {
  const Twine t = Twine("a") + "b" + Twine(42);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: std::string t = (Twine("a") + "b" + Twine(42)).str();
  foo(Twine("a") + "b");

  Twine Prefix = false ? "__INT_FAST" : "__UINT_FAST";
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: const char * Prefix = false ? "__INT_FAST" : "__UINT_FAST";

  const Twine t2 = Twine();
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: std::string t2 = (Twine()).str();
  foo(Twine() + "b");

  const Twine t3 = Twine(42);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: std::string t3 = (Twine(42)).str();

  const Twine t4 = Twine(42) + "b";
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: std::string t4 = (Twine(42) + "b").str();

  const Twine t5 = Twine() + "b";
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: std::string t5 = (Twine() + "b").str();

  const Twine t6 = true ? Twine() : Twine(42);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: std::string t6 = (true ? Twine() : Twine(42)).str();

  const Twine t7 = false ? Twine() : Twine("b");
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: std::string t7 = (false ? Twine() : Twine("b")).str();
}

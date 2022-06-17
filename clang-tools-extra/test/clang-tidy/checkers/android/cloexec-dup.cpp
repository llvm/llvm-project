// RUN: %check_clang_tidy %s android-cloexec-dup %t

extern "C" int dup(int oldfd);
void f() {
  dup(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer fcntl() to dup() because fcntl() allows F_DUPFD_CLOEXEC [android-cloexec-dup]
  // CHECK-FIXES: fcntl(1, F_DUPFD_CLOEXEC);
  int oldfd = 0;
  dup(oldfd);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer
  // CHECK-FIXES: fcntl(oldfd, F_DUPFD_CLOEXEC);
}

namespace i {
int dup(int oldfd);
void g() {
  dup(0);
  int oldfd = 1;
  dup(oldfd);
}
} // namespace i

class C {
public:
  int dup(int oldfd);
  void h() {
    dup(0);
    int oldfd = 1;
    dup(oldfd);
  }
};

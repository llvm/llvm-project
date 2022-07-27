// RUN: %check_clang_tidy %s android-cloexec-pipe %t

extern "C" int pipe(int pipefd[2]);

void warning() {
  int pipefd[2];
  pipe(pipefd);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer pipe2() with O_CLOEXEC to avoid leaking file descriptors to child processes [android-cloexec-pipe]
  // CHECK-FIXES: pipe2(pipefd, O_CLOEXEC);
}

namespace i {
int pipe(int pipefd[2]);
void noWarningInNamespace() {
  int pipefd[2];
  pipe(pipefd);
}
} // namespace i

class C {
public:
  int pipe(int pipefd[2]);
  void noWarningForMemberFunction() {
    int pipefd[2];
    pipe(pipefd);
  }
};

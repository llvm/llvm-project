// RUN: %check_clang_tidy %s android-cloexec-pipe2 %t

#define O_NONBLOCK 1
#define __O_CLOEXEC 3
#define O_CLOEXEC __O_CLOEXEC
#define TEMP_FAILURE_RETRY(exp) \
  ({                            \
    int _rc;                    \
    do {                        \
      _rc = (exp);              \
    } while (_rc == -1);        \
  })
#define NULL 0

extern "C" int pipe2(int pipefd[2], int flags);

void warning() {
  int pipefd[2];
  pipe2(pipefd, O_NONBLOCK);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'pipe2'
  // CHECK-FIXES: pipe2(pipefd, O_NONBLOCK | O_CLOEXEC);
  TEMP_FAILURE_RETRY(pipe2(pipefd, O_NONBLOCK));
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: 'pipe2'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(pipe2(pipefd, O_NONBLOCK | O_CLOEXEC));
}

void warningInMacroArugment() {
  int pipefd[2];
  pipe2(pipefd, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'pipe2'
  // CHECK-FIXES: pipe2(pipefd, 3 | O_CLOEXEC);
  TEMP_FAILURE_RETRY(pipe2(pipefd, 3));
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: 'pipe2'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(pipe2(pipefd, 3 | O_CLOEXEC));

  int flag = O_NONBLOCK;
  pipe2(pipefd, flag);
  TEMP_FAILURE_RETRY(pipe2(pipefd, flag));
}

namespace i {
int pipe2(int pipefd[2], int flags);

void noWarning() {
  int pipefd[2];
  pipe2(pipefd, O_NONBLOCK);
  TEMP_FAILURE_RETRY(pipe2(pipefd, O_NONBLOCK));
}

} // namespace i

void noWarning() {
  int pipefd[2];
  pipe2(pipefd, O_CLOEXEC);
  TEMP_FAILURE_RETRY(pipe2(pipefd, O_CLOEXEC));
  pipe2(pipefd, O_NONBLOCK | O_CLOEXEC);
  TEMP_FAILURE_RETRY(pipe2(pipefd, O_NONBLOCK | O_CLOEXEC));
}

class G {
public:
  int pipe2(int pipefd[2], int flags);
  void noWarning() {
    int pipefd[2];
    pipe2(pipefd, O_NONBLOCK);
    TEMP_FAILURE_RETRY(pipe2(pipefd, O_NONBLOCK));
  }
};

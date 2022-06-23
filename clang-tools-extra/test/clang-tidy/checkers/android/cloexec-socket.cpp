// RUN: %check_clang_tidy %s android-cloexec-socket %t

#define SOCK_STREAM 1
#define SOCK_DGRAM 2
#define __O_CLOEXEC 3
#define SOCK_CLOEXEC __O_CLOEXEC
#define TEMP_FAILURE_RETRY(exp) \
  ({                            \
    int _rc;                    \
    do {                        \
      _rc = (exp);              \
    } while (_rc == -1);        \
  })

extern "C" int socket(int domain, int type, int protocol);

void a() {
  socket(0, SOCK_STREAM, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'socket' should use SOCK_CLOEXEC where possible [android-cloexec-socket]
  // CHECK-FIXES: socket(0, SOCK_STREAM | SOCK_CLOEXEC, 0)
  TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM, 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: 'socket'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM | SOCK_CLOEXEC, 0))
  socket(0, SOCK_STREAM | SOCK_DGRAM, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: 'socket'
  // CHECK-FIXES: socket(0, SOCK_STREAM | SOCK_DGRAM | SOCK_CLOEXEC, 0)
  TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM | SOCK_DGRAM, 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: 'socket'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM | SOCK_DGRAM | SOCK_CLOEXEC, 0))
}

void f() {
  socket(0, 3, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'socket'
  // CHECK-FIXES: socket(0, 3 | SOCK_CLOEXEC, 0)
  TEMP_FAILURE_RETRY(socket(0, 3, 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: 'socket'
  // CHECK-FIXES: TEMP_FAILURE_RETRY(socket(0, 3 | SOCK_CLOEXEC, 0))

  int flag = 3;
  socket(0, flag, 0);
  TEMP_FAILURE_RETRY(socket(0, flag, 0));
}

namespace i {
int socket(int domain, int type, int protocol);

void d() {
  socket(0, SOCK_STREAM, 0);
  TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM, 0));
  socket(0, SOCK_STREAM | SOCK_DGRAM, 0);
  TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM | SOCK_DGRAM, 0));
}

} // namespace i

void e() {
  socket(0, SOCK_CLOEXEC, 0);
  TEMP_FAILURE_RETRY(socket(0, SOCK_CLOEXEC, 0));
  socket(0, SOCK_STREAM | SOCK_CLOEXEC, 0);
  TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM | SOCK_CLOEXEC, 0));
  socket(0, SOCK_STREAM | SOCK_CLOEXEC | SOCK_DGRAM, 0);
  TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM | SOCK_CLOEXEC | SOCK_DGRAM, 0));
}

class G {
public:
  int socket(int domain, int type, int protocol);
  void d() {
    socket(0, SOCK_STREAM, 0);
    TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM, 0));
    socket(0, SOCK_STREAM | SOCK_DGRAM, 0);
    TEMP_FAILURE_RETRY(socket(0, SOCK_STREAM | SOCK_DGRAM, 0));
  }
};

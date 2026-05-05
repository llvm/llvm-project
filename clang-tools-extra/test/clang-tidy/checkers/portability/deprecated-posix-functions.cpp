// RUN: %check_clang_tidy -std=c++98-or-later %s portability-deprecated-posix-functions %t

typedef __SIZE_TYPE__ size_t;
typedef int pid_t;
typedef int uid_t;

int bcmp(const void *S1, const void *S2, size_t N);
void bcopy(const void *Src, void *Dest, size_t N);
void bzero(void *S, size_t N);
int getpw(uid_t UId, char *Buf);
pid_t vfork(void);

#define CALL_BZERO(Buffer, Size) bzero(Buffer, Size)
#define DEPRECATED_BCMP bcmp

void deprecated_posix_functions() {
  char Buf1[128] = {0};
  char Buf2[128] = {0};

  bcmp(Buf1, Buf2, sizeof(Buf1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'bcmp' is deprecated; 'memcmp' should be used instead [portability-deprecated-posix-functions]

  bcopy(Buf1, Buf2, sizeof(Buf1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'bcopy' is deprecated; 'memmove' should be used instead

  bzero(Buf1, sizeof(Buf1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'bzero' is deprecated; 'memset' should be used instead

  getpw(0, Buf1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'getpw' is deprecated; 'getpwuid' should be used instead

  vfork();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'vfork' is deprecated; 'posix_spawn' should be used instead

  int (*BcmpPtr)(const void *, const void *, size_t) = bcmp;
  // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: function 'bcmp' is deprecated; 'memcmp' should be used instead

  CALL_BZERO(Buf1, sizeof(Buf1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'bzero' is deprecated; 'memset' should be used instead

  DEPRECATED_BCMP(Buf1, Buf2, sizeof(Buf1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'bcmp' is deprecated; 'memcmp' should be used instead

  ::bcmp(Buf1, Buf2, sizeof(Buf1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'bcmp' is deprecated; 'memcmp' should be used instead
}

void deprecated_posix_functions_in_control_flow(bool Flag) {
  char Buf1[128] = {0};
  char Buf2[128] = {0};

  if (bcmp(Buf1, Buf2, sizeof(Buf1)) == 0)
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:7: warning: function 'bcmp' is deprecated; 'memcmp' should be used instead

  for (int I = 0; I < 2; ++I)
    bzero(Buf1, sizeof(Buf1));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'bzero' is deprecated; 'memset' should be used instead

  while (getpw(0, Buf1) == 0)
    break;
  // CHECK-MESSAGES: :[[@LINE-2]]:10: warning: function 'getpw' is deprecated; 'getpwuid' should be used instead

  Flag ? bcopy(Buf1, Buf2, sizeof(Buf1)) : bzero(Buf2, sizeof(Buf2));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: function 'bcopy' is deprecated; 'memmove' should be used instead
  // CHECK-MESSAGES: :[[@LINE-2]]:44: warning: function 'bzero' is deprecated; 'memset' should be used instead

  if (Flag && vfork() == 0)
    return;
  // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: function 'vfork' is deprecated; 'posix_spawn' should be used instead
}

namespace non_posix {
int bcmp(const void *S1, const void *S2, size_t N);
void bcopy(const void *Src, void *Dest, size_t N);
void bzero(void *S, size_t N);
int getpw(uid_t UId, char *Buf);
pid_t vfork(void);

void same_unqualified_names() {
  char Buf1[128] = {0};
  char Buf2[128] = {0};

  // No warnings for functions with the same unqualified names outside the global namespace.
  bcmp(Buf1, Buf2, sizeof(Buf1));
  bcopy(Buf1, Buf2, sizeof(Buf1));
  bzero(Buf1, sizeof(Buf1));
  getpw(0, Buf1);
  vfork();
}
} // namespace non_posix
